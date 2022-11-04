// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class HWC2CHWImpl : public HWC2CHW {
 public:
  using HWC2CHW::HWC2CHW;

  Result<Tensor> hwc2chw(const Tensor& img) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device(), stream()));

    SyncOnScopeExit(stream(), src_tensor.buffer() != img.buffer(), src_tensor);

    auto shape = src_tensor.shape();
    int height = shape[1];
    int width = shape[2];
    int channels = shape[3];

    auto dst_mat = mmdeploy::cpu::Transpose(mmdeploy::cpu::Tensor2CVMat(src_tensor));

    auto dst_tensor = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    dst_tensor.Reshape({1, channels, height, width});

    return dst_tensor;
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(HWC2CHW, (cpu, 0), [](const Context& context) {
  return std::make_unique<HWC2CHWImpl>(context);
});

}  // namespace mmdeploy::operation::cpu
