// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

template <typename T>
void Transpose(const T* src, int height, int width, int channels, T* dst, cudaStream_t stream);

class HWC2CHWImpl : public HWC2CHW {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst) override {
    auto h = src.shape(1);
    auto w = src.shape(2);
    auto c = src.shape(3);

    Tensor dst_tensor(src.desc());
    dst_tensor.Reshape({1, c, h, w});

    auto cuda_stream = GetNative<cudaStream_t>(stream());

    if (DataType::kINT8 == src.data_type()) {
      auto input = src.data<uint8_t>();
      auto output = dst_tensor.data<uint8_t>();
      Transpose(input, (int)h, (int)w, (int)c, output, cuda_stream);
    } else if (DataType::kFLOAT == src.data_type()) {
      auto input = src.data<float>();
      auto output = dst_tensor.data<float>();
      Transpose(input, (int)h, (int)w, (int)c, output, cuda_stream);
    } else {
      assert(0);
    }

    dst = std::move(dst_tensor);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(HWC2CHW, (cuda, 0), [] { return std::make_unique<HWC2CHWImpl>(); });

}  // namespace mmdeploy::operation::cuda
