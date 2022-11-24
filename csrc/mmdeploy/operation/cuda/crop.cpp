// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

namespace impl {

template <typename T, int channels>
void Crop(const T* src, int src_w, T* dst, int dst_h, int dst_w, int offset_h, int offset_w,
          cudaStream_t stream);

}

class CropImpl : public Crop {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst, int top, int left, int bottom,
                     int right) override {
    auto cuda_stream = GetNative<cudaStream_t>(stream());
    auto desc = src.desc();

    int h = bottom - top + 1;
    int w = right - left + 1;
    int c = desc.shape[3];
    auto type = desc.data_type;

    TensorShape shape{1, bottom - top + 1, right - left + 1, src.desc().shape[3]};
    TensorDesc dst_desc{device(), src.desc().data_type, shape, desc.name};
    Tensor dst_tensor{dst_desc};

    if (DataType::kINT8 == type) {
      auto input = src.data<uint8_t>();
      auto output = dst_tensor.data<uint8_t>();
      if (3 == c) {
        impl::Crop<uint8_t, 3>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else if (1 == c) {
        impl::Crop<uint8_t, 1>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else if (DataType::kFLOAT == type) {
      auto input = static_cast<float*>(src.buffer().GetNative());
      auto output = static_cast<float*>(dst_tensor.buffer().GetNative());
      if (3 == c) {
        impl::Crop<float, 3>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else if (1 == c) {
        impl::Crop<float, 1>(input, desc.shape[2], output, h, w, top, left, cuda_stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else {
      MMDEPLOY_ERROR("unsupported type {}", type);
      return Status(eNotSupported);
    }

    dst = std::move(dst_tensor);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Crop, (cuda, 0), [] { return std::make_unique<CropImpl>(); });

}  // namespace mmdeploy::operation::cuda
