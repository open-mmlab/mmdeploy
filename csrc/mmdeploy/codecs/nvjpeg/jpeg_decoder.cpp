// Copyright (c) OpenMMLab. All rights reserved.

#include "jpeg_decoder.h"

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <map>
#include <mutex>
#include <vector>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/archive.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/device_impl.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/value.h"

#define CHECK_NVJPEG(call)                                                      \
  {                                                                             \
    nvjpegStatus_t _e = (call);                                                 \
    if (_e != NVJPEG_STATUS_SUCCESS) {                                          \
      MMDEPLOY_ERROR("NVJPEG failure: '#{}' at {} {}", _e, __FILE__, __LINE__); \
      throw std::runtime_error("NVJPEG failure");                               \
    }                                                                           \
  }

namespace mmdeploy {

namespace codecs {

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

struct JPEGDecoder::Impl {
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

  Impl(int device_id) {
    device_ = Device("cuda", device_id);
    auto stream = Stream::GetDefault(device_);
    stream_ = GetNative<cudaStream_t>(stream);

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
  }

  ~Impl() {
    try {
      CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams_[0]));
      CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams_[1]));
      CHECK_NVJPEG(nvjpegJpegStateDestroy(state_));
      CHECK_NVJPEG(nvjpegDecoderDestroy(decoder_));
      CHECK_NVJPEG(nvjpegJpegStateDestroy(decoupled_state_));
      CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers_[0]));
      CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers_[1]));
      CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer_));
      CHECK_NVJPEG(nvjpegDecodeParamsDestroy(decode_params_));
      CHECK_NVJPEG(nvjpegDestroy(handle_));  // destroy last
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("JPEGDecoder doesn't deconstruct properly");
    }
  }

  void Prepare(const std::vector<const char*>& raw_data, const std::vector<int>& length,
               PixelFormat format, std::vector<const unsigned char*>& batched_data,
               std::vector<size_t>& batched_len, std::vector<nvjpegImage_t>& batched_nv_images,
               std::vector<const unsigned char*>& normal_data, std::vector<size_t>& normal_len,
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
  }

  Result<Value> Apply(const std::vector<const char*>& raw_data, const std::vector<int>& length,
                      PixelFormat format) {
    if (format != PixelFormat::kBGR && format != PixelFormat::kRGB ||
        raw_data.size() != length.size()) {
      return Status(eInvalidArgument);
    }

    std::vector<const unsigned char*> batched_data;
    std::vector<size_t> batched_len;
    std::vector<nvjpegImage_t> batched_nv_images;
    std::vector<const unsigned char*> normal_data;
    std::vector<size_t> normal_len;
    std::vector<nvjpegImage_t> normal_nv_images;
    std::vector<Mat> mats;

    Prepare(raw_data, length, format, batched_data, batched_len, batched_nv_images, normal_data,
            normal_len, normal_nv_images, mats);

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
        cudaStreamSynchronize(stream_);

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, decoder_, decoupled_state_,
                                                      jpeg_streams_[buffer_index], stream_));
        buffer_index = 1 - buffer_index;
        CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, decoder_, decoupled_state_,
                                            &normal_nv_images[i], stream_));
      }
    }

    return to_value(mats);
  }
};

JPEGDecoder::JPEGDecoder(int device_id) { impl_ = std::make_unique<Impl>(device_id); }

Result<Value> JPEGDecoder::Apply(const std::vector<const char*>& raw_data,
                                 const std::vector<int>& length, PixelFormat format) {
  return impl_->Apply(raw_data, length, format);
}

JPEGDecoder::~JPEGDecoder() = default;

}  // namespace codecs
}  // namespace mmdeploy