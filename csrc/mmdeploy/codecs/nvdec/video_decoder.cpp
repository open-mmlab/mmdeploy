// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codecs/nvdec/video_decoder.h"

#include <cuda.h>
#include <cuviddec.h>
#include <nppcore.h>
#include <nppi_color_conversion.h>
#include <nvcuvid.h>

#include <thread>
#include <vector>

#include "mmdeploy/codecs/nvdec/frame_queue.h"
#include "mmdeploy/codecs/nvdec/video_demuxer.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/status_code.h"

#define NVDEC_CHECK(call) nvdec_check(call, __LINE__, __FILE__)

namespace mmdeploy {

namespace nvdec {

void nvdec_check(CUresult err, int line, const char* file) {
  if (err != 0) {
    const char* msg = nullptr;
    cuGetErrorString(err, &msg);
    MMDEPLOY_ERROR("NVDEC: {} {} {}", line, file, msg);
    throw_exception(eFail);
  }
}

using rect_t = decltype(CUVIDEOFORMAT::display_area);
bool operator==(const rect_t& a1, const rect_t& a2) {
  return (a1.left == a2.left) && (a1.top == a2.top) && (a1.right == a2.right) &&
         (a1.bottom == a2.bottom);
}

struct VideoDecoder::Impl {
  int device_id_;
  framework::Device device_;
  std::string source_;
  PixelFormat format_{PixelFormat::kBGR};

  std::mutex mutex_{};
  CUcontext ctx_{nullptr};
  CUstream stream_{nullptr};
  CUvideoctxlock lock_{nullptr};
  CUvideodecoder decoder_{nullptr};
  CUvideoparser parser_{nullptr};
  std::shared_ptr<FrameQueue> queue_;
  std::shared_ptr<VideoDemuxer> demuxer_;
  CUVIDEOFORMAT video_format_;
  std::shared_ptr<std::thread> read_thread_;
  bool loop_running_{false};
  std::vector<CUVIDPARSERDISPINFO> parsed_disp_info_;

  Result<void> Init(const Value& args) {
    device_id_ = args["device_id"].get<int>();
    source_ = args["path"].get<const char*>();
    device_ = framework::Device("cuda", device_id_);
    format_ = PixelFormat(args["pixel_format"].get<int>());
    demuxer_ = std::make_shared<VideoDemuxer>(source_);
    if (format_ != PixelFormat::kBGR && format_ != PixelFormat::kRGB) {
      MMDEPLOY_ERROR("Only support BGR or RGB format");
      throw_exception(eNotSupported);
    }

    CUdevice cuDevice = 0;
    NVDEC_CHECK(cuDeviceGet(&cuDevice, device_id_));
    NVDEC_CHECK(cuCtxCreate(&ctx_, 0, cuDevice));
    NVDEC_CHECK(cuvidCtxLockCreate(&lock_, ctx_));
    NVDEC_CHECK(cuvidCtxLock(lock_, 0));
    NVDEC_CHECK(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
    NVDEC_CHECK(cuvidCtxUnlock(lock_, 0));

    CUVIDPARSERPARAMS parser_params = {};
    parser_params.CodecType = demuxer_->CodecType();
    parser_params.ulMaxNumDecodeSurfaces = 1;
    parser_params.ulMaxDisplayDelay = 0;
    parser_params.pUserData = this;
    parser_params.pfnSequenceCallback = HandleVideoSequence;
    parser_params.pfnDecodePicture = HandlePictureDecode;
    parser_params.pfnDisplayPicture = HandlePictureDisplay;
    parser_params.pfnGetOperatingPoint = HandleOperatingPoint;
    NVDEC_CHECK(cuvidCreateVideoParser(&parser_, &parser_params));
    StartReadLoop();
    return success();
  }

  void StartReadLoop() {
    loop_running_ = true;
    read_thread_.reset(new std::thread(Loop, this));
  }

  void StopReadLoop() {
    loop_running_ = false;
    if (read_thread_) {
      read_thread_->join();
      read_thread_.reset();
    }
  }

  static void Loop(void* user_data) {
    Impl* _this = static_cast<Impl*>(user_data);

    while (_this->loop_running_) {
      unsigned char* data = nullptr;
      size_t size = 0;

      auto ok = _this->demuxer_->GetNextPacket(&data, &size);

      CUVIDSOURCEDATAPACKET packet = {};
      packet.payload_size = size;
      packet.payload = data;
      if (!data || size == 0) {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
      }
      NVDEC_CHECK(cuvidParseVideoData(_this->parser_, &packet));

      if (!ok) {
        _this->queue_->EndDecode();
        break;
      }

      if (!_this->loop_running_) {
        break;
      }
    }
  }

  Result<void> GetInfo(VideoInfo& info) {
    if (demuxer_) {
      return demuxer_->GetInfo(info);
    }
    return Status(eFail);
  }

  Result<void> Read(framework::Mat& out) {
    if (Grab()) {
      return Retrieve(out);
    }
    return Status(eFail);
  }

  Result<void> Retrieve(framework::Mat& out) {
    if (parsed_disp_info_.empty()) {
      return Status(eFail);
    }

    CUVIDPARSERDISPINFO disp_info = parsed_disp_info_.front();
    parsed_disp_info_.clear();

    CUVIDPROCPARAMS proc_params = {};
    proc_params.progressive_frame = disp_info.progressive_frame;
    proc_params.second_field = disp_info.repeat_first_field + 1;
    proc_params.top_field_first = disp_info.top_field_first;
    proc_params.unpaired_field = disp_info.repeat_first_field < 0;
    proc_params.output_stream = stream_;
    NVDEC_CHECK(cuvidCtxLock(lock_, 0));
    CUdeviceptr ptr;
    unsigned int pitch;
    NVDEC_CHECK(cuvidMapVideoFrame(decoder_, disp_info.picture_index, &ptr, &pitch, &proc_params));

    int width = video_format_.display_area.right - video_format_.display_area.left;
    int height = video_format_.display_area.bottom - video_format_.display_area.top;
    framework::Mat _out(height, width, format_, DataType::kINT8, framework::Device("cuda"));
    auto stream = framework::Stream(device_, stream_);

    // color convert
    Npp8u* pSrc[2] = {(unsigned char*)ptr, (unsigned char*)ptr + height * pitch};
    NppiSize oSizeROI = {width, height};
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream_;
    NppStatus status;
    if (format_ == PixelFormat::kBGR) {
      status = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, pitch, _out.data<unsigned char>(), width * 3,
                                          oSizeROI, nppStreamCtx);
    } else if (format_ == PixelFormat::kRGB) {
      status = nppiNV12ToRGB_8u_P2C3R_Ctx(pSrc, pitch, _out.data<unsigned char>(), width * 3,
                                          oSizeROI, nppStreamCtx);
    }

    OUTCOME_TRY(stream.Wait());
    NVDEC_CHECK(cuvidUnmapVideoFrame(decoder_, ptr));
    NVDEC_CHECK(cuvidCtxUnlock(lock_, 0));
    queue_->ReleaseFrame(&disp_info);
    if (status != NPP_SUCCESS) {
      return Status(eFail);
    }
    out = _out;
    return success();
  }

  Result<void> Grab() {
    // drop last frame
    if (!parsed_disp_info_.empty()) {
      queue_->ReleaseFrame(&parsed_disp_info_.front());
      parsed_disp_info_.clear();
    }

    CUVIDPARSERDISPINFO disp_info;
    while (true) {
      if (queue_ && queue_->Dequeue(&disp_info)) {
        break;
      }
      if (queue_ && queue_->IsEnd()) {
        return Status(eFail);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    parsed_disp_info_.push_back(disp_info);
    return success();
  };

  ~Impl() {
    StopReadLoop();

    if (!ctx_) {
      return;
    }
    if (parser_) {
      cuvidDestroyVideoParser(parser_);
    }
    if (lock_) {
      cuvidCtxLock(lock_, 0);
      if (decoder_) {
        cuvidDestroyDecoder(decoder_);
      }
      cuvidCtxUnlock(lock_, 0);
      cuvidCtxLockDestroy(lock_);
    }
    cuCtxDestroy(ctx_);
  }

  static int CUDAAPI HandleVideoSequence(void* user_data, CUVIDEOFORMAT* video_format) {
    Impl* _this = static_cast<Impl*>(user_data);

    // Check video format is supported by GPU's hardware video decoder
    CUVIDDECODECAPS caps{};
    caps.eCodecType = video_format->codec;
    caps.eChromaFormat = video_format->chroma_format;
    caps.nBitDepthMinus8 = video_format->bit_depth_luma_minus8;
    NVDEC_CHECK(cuvidCtxLock(_this->lock_, 0));
    NVDEC_CHECK(cuvidGetDecoderCaps(&caps));
    NVDEC_CHECK(cuvidCtxUnlock(_this->lock_, 0));

    if (!caps.bIsSupported) {
      MMDEPLOY_ERROR("Codec {} not supported on this GPU", (int)video_format->codec);
      throw_exception(eNotSupported);
    }
    if (!(caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))) {
      MMDEPLOY_ERROR("Format {} not supported", (int)caps.nOutputFormatMask);
      throw_exception(eNotSupported);
    }
    if (video_format->coded_width < caps.nMinWidth || video_format->coded_width > caps.nMaxWidth) {
      MMDEPLOY_ERROR("Video width {} not in range ({} {})", video_format->coded_width,
                     caps.nMinWidth, caps.nMaxWidth);
      throw_exception(eNotSupported);
    }
    if (video_format->coded_height < caps.nMinHeight ||
        video_format->coded_height > caps.nMaxHeight) {
      MMDEPLOY_ERROR("Video height {} not in range ({} {})", video_format->coded_height,
                     caps.nMinHeight, caps.nMaxHeight);
      throw_exception(eNotSupported);
    }
    if (video_format->coded_width * video_format->coded_height / 256 > caps.nMaxMBCount) {
      MMDEPLOY_ERROR("MBCount {} exceed {}",
                     video_format->coded_width * video_format->coded_height / 256,
                     caps.nMaxMBCount);
      throw_exception(eNotSupported);
    }

    // reconfigure decoder
    if (_this->queue_) {
      CUVIDEOFORMAT* prev_format = &_this->video_format_;
      if (prev_format->bit_depth_luma_minus8 != video_format->bit_depth_luma_minus8 ||
          prev_format->bit_depth_chroma_minus8 != video_format->bit_depth_chroma_minus8 ||
          prev_format->chroma_format != video_format->chroma_format) {
        MMDEPLOY_ERROR("Reconfigure not supported for bit depth or chroma change");
        throw_exception(eNotSupported);
      }

      bool size_change = (prev_format->coded_width != video_format->coded_width) ||
                         (prev_format->coded_height != video_format->coded_height);
      bool disp_rect_change = !(prev_format->display_area == video_format->display_area);

      if (size_change || disp_rect_change) {
        CUVIDRECONFIGUREDECODERINFO reconfig_info;
        reconfig_info.ulWidth = video_format->coded_width;
        reconfig_info.ulHeight = video_format->coded_height;
        reconfig_info.ulTargetWidth =
            video_format->display_area.right - video_format->display_area.left;
        reconfig_info.ulTargetHeight =
            video_format->display_area.bottom - video_format->display_area.top;
        reconfig_info.display_area.left = video_format->display_area.left;
        reconfig_info.display_area.top = video_format->display_area.top;
        reconfig_info.display_area.right = video_format->display_area.right;
        reconfig_info.display_area.bottom = video_format->display_area.bottom;
        reconfig_info.ulNumDecodeSurfaces = video_format->min_num_decode_surfaces;
        *prev_format = *video_format;
        NVDEC_CHECK(cuvidCtxLock(_this->lock_, 0));
        NVDEC_CHECK(cuvidReconfigureDecoder(_this->decoder_, &reconfig_info));
        NVDEC_CHECK(cuvidCtxUnlock(_this->lock_, 0));
        return reconfig_info.ulNumDecodeSurfaces;
      }
      return 1;
    }

    // init decoder
    _this->video_format_ = *video_format;

    _this->queue_ = std::make_shared<FrameQueue>();
    _this->queue_->Init(32);  // max value of ulNumDecodeSurfaces

    CUVIDDECODECREATEINFO info{};
    info.ulWidth = video_format->coded_width;
    info.ulHeight = video_format->coded_height;
    info.ulNumDecodeSurfaces = video_format->min_num_decode_surfaces;
    info.CodecType = video_format->codec;
    info.ChromaFormat = video_format->chroma_format;
    info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    info.bitDepthMinus8 = video_format->bit_depth_luma_minus8;
    info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    info.DeinterlaceMode = video_format->progressive_sequence ? cudaVideoDeinterlaceMode_Weave
                                                              : cudaVideoDeinterlaceMode_Adaptive;
    info.ulNumOutputSurfaces = 2;
    info.vidLock = _this->lock_;
    int max_width = 0, max_height = 0;
    if (video_format->codec == cudaVideoCodec_AV1 && video_format->seqhdr_data_length > 0) {
      CUVIDEOFORMATEX* video_format_ex = (CUVIDEOFORMATEX*)video_format;
      max_width = video_format_ex->av1.max_width;
      max_height = video_format_ex->av1.max_height;
    }
    if (max_width < video_format->coded_width) {
      max_width = video_format->coded_width;
    }
    if (max_height < video_format->coded_height) {
      max_height = video_format->coded_height;
    }
    info.ulMaxWidth = max_width;
    info.ulMaxHeight = max_height;
    info.ulTargetWidth = video_format->display_area.right - video_format->display_area.left;
    info.ulTargetHeight = video_format->display_area.bottom - video_format->display_area.top;

    NVDEC_CHECK(cuCtxPushCurrent(_this->ctx_));
    NVDEC_CHECK(cuvidCreateDecoder(&_this->decoder_, &info));
    NVDEC_CHECK(cuCtxPopCurrent(NULL));

    return info.ulNumDecodeSurfaces;
  }

  static int CUDAAPI HandlePictureDecode(void* user_data, CUVIDPICPARAMS* pic_params) {
    Impl* _this = static_cast<Impl*>(user_data);
    bool is_frame_available = _this->queue_->WaitUntilFrameAvailable(pic_params->CurrPicIdx);
    if (!is_frame_available) {
      return false;
    }
    NVDEC_CHECK(cuvidDecodePicture(_this->decoder_, pic_params));
    return true;
  }

  static int CUDAAPI HandlePictureDisplay(void* user_data, CUVIDPARSERDISPINFO* disp_info) {
    Impl* _this = static_cast<Impl*>(user_data);
    _this->queue_->Enqueue(disp_info);
    return true;
  }

  static int CUDAAPI HandleOperatingPoint(void* user_data,
                                          CUVIDOPERATINGPOINTINFO* operating_info) {
    Impl* _this = static_cast<Impl*>(user_data);
    if (operating_info->codec == cudaVideoCodec_AV1 &&
        operating_info->av1.operating_points_cnt > 1) {
      return 0;
    }
    return -1;
  }
};

VideoDecoder::VideoDecoder() { impl_ = std::make_unique<Impl>(); }

Result<void> VideoDecoder::Init(const Value& args) { return impl_->Init(args); }

Result<void> VideoDecoder::GetInfo(VideoInfo& info) { return impl_->GetInfo(info); };

Result<void> VideoDecoder::Read(framework::Mat& out) { return impl_->Read(out); }

Result<void> VideoDecoder::Retrieve(framework::Mat& out) { return impl_->Retrieve(out); }

Result<void> VideoDecoder::Grab() { return impl_->Grab(); }

VideoDecoder::~VideoDecoder() = default;

}  // namespace nvdec

static std::unique_ptr<::mmdeploy::VideoDecoder> Create(const Value& args) {
  auto p = std::make_unique<nvdec::VideoDecoder>();
  if (p->Init(args)) {
    return p;
  }
  return nullptr;
}

MMDEPLOY_REGISTER_FACTORY_FUNC(VideoDecoder, (cuda, 0), Create);

}  // namespace mmdeploy
