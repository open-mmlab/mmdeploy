#include "mmdeploy/codecs/nvjpeg/nvjpeg_decoder.h"

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <unordered_map>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codecs/decoder.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/device_impl.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"

void check_nvjpeg(nvjpegStatus_t err, int line, const char* file, bool exit) {
  static std::unordered_map<nvjpegStatus_t, const char*> code2str = {
      {NVJPEG_STATUS_SUCCESS, "NVJPEG_STATUS_SUCCESS"},
      {NVJPEG_STATUS_NOT_INITIALIZED, "NVJPEG_STATUS_NOT_INITIALIZED"},
      {NVJPEG_STATUS_INVALID_PARAMETER, "NVJPEG_STATUS_INVALID_PARAMETER"},
      {NVJPEG_STATUS_BAD_JPEG, "NVJPEG_STATUS_BAD_JPEG"},
      {NVJPEG_STATUS_JPEG_NOT_SUPPORTED, "NVJPEG_STATUS_JPEG_NOT_SUPPORTED"},
      {NVJPEG_STATUS_ALLOCATOR_FAILURE, "NVJPEG_STATUS_ALLOCATOR_FAILURE"},
      {NVJPEG_STATUS_EXECUTION_FAILED, "NVJPEG_STATUS_EXECUTION_FAILED"},
      {NVJPEG_STATUS_ARCH_MISMATCH, "NVJPEG_STATUS_ARCH_MISMATCH"},
      {NVJPEG_STATUS_INTERNAL_ERROR, "NVJPEG_STATUS_INTERNAL_ERROR"},
      {NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED, "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED"}};

  if (err != 0) {
    MMDEPLOY_ERROR("NVJPEG failure: '{}' @ {}:{}", code2str[err], file, line);
    if (exit) {
      mmdeploy::throw_exception(mmdeploy::eFail);
    }
  }
}

#define CHECK_NVJPEG(call) check_nvjpeg(call, __LINE__, __FILE__, true)
#define CHECK_NVJPEG_NT(call) check_nvjpeg(call, __LINE__, __FILE__, false)

namespace mmdeploy {

namespace nvjpeg {

using namespace framework;

struct AllocatorBridge {
  AllocatorBridge(int device_id) {
    device_ = Device("cuda", device_id);
    allocator_ = Buffer(device_, 0).GetAllocator();
  };
  AllocatorBridge(const AllocatorBridge&) = delete;
  AllocatorBridge(AllocatorBridge&&) = delete;
  AllocatorBridge& operator=(const AllocatorBridge&) = delete;
  AllocatorBridge& operator=(AllocatorBridge&&) = delete;

  int DevMalloc(void** p, size_t s) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto block = Access::get<AllocatorImpl>(allocator_).Allocate(s);
    *p = block.handle;
    mp_[block.handle] = block;
    return 0;
  }

  int DevFree(void* p) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto block = mp_[p];
    Access::get<AllocatorImpl>(allocator_).Deallocate(block);
    mp_.erase(p);
    return 0;
  }

  Device device_;
  Allocator allocator_;
  std::mutex mutex_;
  std::map<void*, AllocatorImpl::Block> mp_;
};

template <int device_id>
AllocatorBridge& gAllocatorBridge() {
  static AllocatorBridge instance(device_id);
  return instance;
};

template <int device_id>
int DevMalloc(void** ptr, size_t size) {
  auto& handle = gAllocatorBridge<device_id>();
  return handle.DevMalloc(ptr, size);
}

template <int device_id>
int DevFree(void* ptr) {
  auto& handle = gAllocatorBridge<device_id>();
  return handle.DevFree(ptr);
}

std::vector<tDevMalloc> dev_malloc_funcs;
std::vector<tDevFree> dev_free_funcs;

template <size_t N>
struct RegisterAllocator : RegisterAllocator<N - 1> {
  RegisterAllocator() : RegisterAllocator<N - 1>() {
    dev_malloc_funcs.push_back(DevMalloc<N - 1>);
    dev_free_funcs.push_back(DevFree<N - 1>);
  }
};

template <>
struct RegisterAllocator<0> {};

static RegisterAllocator<64> register_allocator;

tDevMalloc GeDevtMalloc(int device_id) { return dev_malloc_funcs[device_id]; }

tDevFree GetDevFree(int device_id) { return dev_free_funcs[device_id]; }

struct ImageDecoder::Impl {
  Device device_;
  cudaStream_t stream_;
  nvjpegHandle_t handle_;
  nvjpegJpegState_t state_;
  nvjpegJpegState_t decoupled_state_;
  nvjpegJpegStream_t jpeg_streams_[2];
  nvjpegBufferPinned_t pinned_buffers_[2];
  nvjpegBufferDevice_t device_buffer_;
  nvjpegDecodeParams_t decode_params_;
  nvjpegJpegDecoder_t decoder_;

  bool hw_decode_available_;  // A100, A30

  Result<void> Init(const Value& cfg) {
    device_ = cfg["device"].get<Device>();
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    auto device_id = device_.device_id();
    nvjpegDevAllocator_t dev_allocator = {GeDevtMalloc(device_id), GetDevFree(device_id)};
    hw_decode_available_ = true;
    nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator, nullptr,
                                           NVJPEG_FLAGS_DEFAULT, &handle_);
    if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
      MMDEPLOY_WARN("Hardware decoder not supported. Using default backend");
      CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, nullptr,
                                  NVJPEG_FLAGS_DEFAULT, &handle_));
      hw_decode_available_ = false;
    }

    CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &jpeg_streams_[0]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &jpeg_streams_[1]));
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    CHECK_NVJPEG(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_DEFAULT, &decoder_));
    CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, decoder_, &decoupled_state_));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle_, NULL, &pinned_buffers_[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle_, NULL, &pinned_buffers_[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle_, &dev_allocator, &device_buffer_));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &decode_params_));
    return success();
  }

  ~Impl() {
    CHECK_NVJPEG_NT(nvjpegJpegStreamDestroy(jpeg_streams_[0]));
    CHECK_NVJPEG_NT(nvjpegJpegStreamDestroy(jpeg_streams_[1]));
    CHECK_NVJPEG_NT(nvjpegJpegStateDestroy(state_));
    CHECK_NVJPEG_NT(nvjpegDecoderDestroy(decoder_));
    CHECK_NVJPEG_NT(nvjpegJpegStateDestroy(decoupled_state_));
    CHECK_NVJPEG_NT(nvjpegBufferPinnedDestroy(pinned_buffers_[0]));
    CHECK_NVJPEG_NT(nvjpegBufferPinnedDestroy(pinned_buffers_[1]));
    CHECK_NVJPEG_NT(nvjpegBufferDeviceDestroy(device_buffer_));
    CHECK_NVJPEG_NT(nvjpegDecodeParamsDestroy(decode_params_));
    CHECK_NVJPEG_NT(nvjpegDestroy(handle_));  // destroy last
    cudaStreamDestroy(stream_);
  }

  Result<void> Prepare(const std::vector<const char*>& raw_data, const std::vector<int>& length,
                       PixelFormat format, std::vector<const unsigned char*>& batched_data,
                       std::vector<size_t>& batched_len,
                       std::vector<nvjpegImage_t>& batched_nv_images,
                       std::vector<const unsigned char*>& normal_data,
                       std::vector<size_t>& normal_len,
                       std::vector<nvjpegImage_t>& normal_nv_images, std::vector<Mat>& mats) {
    int n_image = raw_data.size();

    for (int i = 0; i < n_image; i++) {
      int components;
      nvjpegChromaSubsampling_t subsampling;
      int widths[NVJPEG_MAX_COMPONENT];
      int heights[NVJPEG_MAX_COMPONENT];
      CHECK_NVJPEG(nvjpegGetImageInfo(handle_, (const unsigned char*)raw_data[i], length[i],
                                      &components, &subsampling, widths, heights));

      int width = widths[0];
      int height = heights[0];

      Mat mat(height, width, format, DataType::kINT8, device_);
      mats.push_back(mat);

      nvjpegImage_t out_image;
      for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
        out_image.channel[c] = nullptr;
        out_image.pitch[c] = 0;
      }

      out_image.channel[0] = mat.data<uint8_t>();
      out_image.pitch[0] = width * 3;

      int supported = -1;
      if (hw_decode_available_) {
        CHECK_NVJPEG(nvjpegJpegStreamParseHeader(handle_, (const unsigned char*)raw_data[i],
                                                 length[i], jpeg_streams_[0]));
        CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, jpeg_streams_[0], &supported));
      }

      if (supported == 0) {
        batched_data.push_back((const unsigned char*)raw_data[i]);
        batched_len.push_back(length[i]);
        batched_nv_images.push_back(out_image);
      } else {
        normal_data.push_back((const unsigned char*)raw_data[i]);
        normal_len.push_back(length[i]);
        normal_nv_images.push_back(out_image);
      }
    }

    return success();
  }

  Result<Value> Process(const std::vector<const char*>& raw_data, const std::vector<int>& length,
                        PixelFormat format) {
    if (format != PixelFormat::kBGR && format != PixelFormat::kRGB ||
        raw_data.size() != length.size()) {
      return Status(eInvalidArgument);
    }

    auto stream = Stream(device_, stream_);

    std::vector<const unsigned char*> batched_data;
    std::vector<size_t> batched_len;
    std::vector<nvjpegImage_t> batched_nv_images;
    std::vector<const unsigned char*> normal_data;
    std::vector<size_t> normal_len;
    std::vector<nvjpegImage_t> normal_nv_images;
    std::vector<Mat> mats;

    OUTCOME_TRY(Prepare(raw_data, length, format, batched_data, batched_len, batched_nv_images,
                        normal_data, normal_len, normal_nv_images, mats));

    nvjpegOutputFormat_t output_format =
        (format == PixelFormat::kBGR) ? NVJPEG_OUTPUT_BGRI : NVJPEG_OUTPUT_RGBI;

    if (batched_data.size() > 0) {
      CHECK_NVJPEG(
          nvjpegDecodeBatchedInitialize(handle_, state_, batched_data.size(), 1, output_format));
      CHECK_NVJPEG(nvjpegDecodeBatched(handle_, state_, batched_data.data(), batched_len.data(),
                                       batched_nv_images.data(), stream_));
    }

    if (normal_data.size() > 0) {
      CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(decoupled_state_, device_buffer_));
      CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decode_params_, output_format));

      int buffer_index = 0;
      for (int i = 0; i < normal_data.size(); i++) {
        CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, normal_data[i], normal_len[i], 0, 0,
                                           jpeg_streams_[buffer_index]));

        CHECK_NVJPEG(
            nvjpegStateAttachPinnedBuffer(decoupled_state_, pinned_buffers_[buffer_index]));
        CHECK_NVJPEG(nvjpegDecodeJpegHost(handle_, decoder_, decoupled_state_, decode_params_,
                                          jpeg_streams_[buffer_index]));
        OUTCOME_TRY(stream.Wait());

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, decoder_, decoupled_state_,
                                                      jpeg_streams_[buffer_index], stream_));
        buffer_index = 1 - buffer_index;
        CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, decoder_, decoupled_state_,
                                            &normal_nv_images[i], stream_));
      }
    }

    OUTCOME_TRY(stream.Wait());
    return to_value(mats);
  }

  Result<Value> Process(const Value& input) {
    auto _input = from_value<std::vector<ImageDecoderInput>>(input["input"]);
    std::vector<const char*> raw_data;
    std::vector<int> length;
    PixelFormat format = PixelFormat::kBGR;

    for (int i = 0; i < _input.size(); i++) {
      raw_data.push_back(_input[i].raw_data);
      length.push_back(_input[i].length);
      format = _input[i].format;
    }
    return Process(raw_data, length, format);
  }
};

ImageDecoder::ImageDecoder() { impl_ = std::make_unique<Impl>(); }

Result<void> ImageDecoder::Init(const Value& cfg) { return impl_->Init(cfg); }

Result<Value> ImageDecoder::Process(const Value& input) { return impl_->Process(input); }

ImageDecoder::~ImageDecoder() = default;

}  // namespace nvjpeg

static std::unique_ptr<::mmdeploy::ImageDecoder> Create(const Value& args) {
  auto p = std::make_unique<nvjpeg::ImageDecoder>();
  if (p->Init(args)) {
    return p;
  }
  return nullptr;
}

MMDEPLOY_REGISTER_FACTORY_FUNC(ImageDecoder, (cuda, 0), Create);

}  // namespace mmdeploy
