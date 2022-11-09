// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

namespace impl {
template <typename T, int channels>
void Normalize(const T* src, int height, int width, int stride, float* output, const float* mean,
               const float* std, bool to_rgb, cudaStream_t stream);
}

class NormalizeImpl : public Normalize {
 public:
  NormalizeImpl(Param param) : param_(std::move(param)) {}

  Result<void> apply(const Tensor& src, Tensor& dst) override {
    auto src_desc = src.desc();
    int h = (int)src_desc.shape[1];
    int w = (int)src_desc.shape[2];
    int c = (int)src_desc.shape[3];
    int stride = w * c;

    TensorDesc dst_desc{device(), DataType::kFLOAT, src_desc.shape, src_desc.name};
    Tensor dst_tensor{dst_desc};
    auto output = dst_tensor.data<float>();
    auto cuda_stream = GetNative<cudaStream_t>(stream());

    if (DataType::kINT8 == src_desc.data_type) {
      auto input = src.data<uint8_t>();
      if (3 == c) {
        impl::Normalize<uint8_t, 3>(input, h, w, stride, output, param_.mean.data(),
                                    param_.std.data(), param_.to_rgb, cuda_stream);
      } else if (1 == c) {
        impl::Normalize<uint8_t, 1>(input, h, w, stride, output, param_.mean.data(),
                                    param_.std.data(), param_.to_rgb, cuda_stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else if (DataType::kFLOAT == src_desc.data_type) {
      auto input = src.data<float>();
      if (3 == c) {
        impl::Normalize<float, 3>(input, h, w, stride, output, param_.mean.data(),
                                  param_.std.data(), param_.to_rgb, cuda_stream);
      } else if (1 == c) {
        impl::Normalize<float, 1>(input, h, w, stride, output, param_.mean.data(),
                                  param_.std.data(), param_.to_rgb, cuda_stream);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", src_desc.data_type);
      assert(0);
      return Status(eNotSupported);
    }

    dst = std::move(dst_tensor);
    return success();
  }

 protected:
  Param param_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Normalize, (cuda, 0), [](const Normalize::Param& param) {
  return std::make_unique<NormalizeImpl>(param);
});

}  // namespace mmdeploy::operation::cuda
