// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"

namespace mmdeploy::operation::cuda {

template <typename T>
void Transpose(const T* src, int height, int width, int channels, T* dst, cudaStream_t stream);

class HWC2CHWImpl : public HWC2CHW {
 public:
  using HWC2CHW::HWC2CHW;

  Result<Tensor> hwc2chw(const Tensor& img) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device(), stream()));

    SyncOnScopeExit sync(stream(), src_tensor.buffer() != img.buffer(), src_tensor);

    auto h = img.shape(1);
    auto w = img.shape(2);
    auto c = img.shape(3);
    auto hw = h * w;

    Tensor dst_tensor(src_tensor.desc());
    dst_tensor.Reshape({1, c, h, w});

    auto cuda_stream = GetNative<cudaStream_t>(stream());

    if (DataType::kINT8 == img.data_type()) {
      auto input = src_tensor.data<uint8_t>();
      auto output = dst_tensor.data<uint8_t>();
      Transpose(input, (int)h, (int)w, (int)c, output, cuda_stream);
    } else if (DataType::kFLOAT == img.data_type()) {
      auto input = src_tensor.data<float>();
      auto output = dst_tensor.data<float>();
      Transpose(input, (int)h, (int)w, (int)c, output, cuda_stream);
    } else {
      assert(0);
    }
    return dst_tensor;
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(HWC2CHW, (cuda, 0), [](const Context& context) {
  return std::make_unique<HWC2CHWImpl>(context);
});

}  // namespace mmdeploy::operation::cuda
