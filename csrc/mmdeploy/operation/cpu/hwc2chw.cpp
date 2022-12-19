// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class HWC2CHWImpl : public HWC2CHW {
 public:
  Result<void> apply(const Tensor& img, Tensor& dst) override {
    auto shape = img.shape();
    auto height = shape[1];
    auto width = shape[2];
    auto channels = shape[3];

    auto dst_mat = mmdeploy::cpu::Transpose(mmdeploy::cpu::Tensor2CVMat(img));

    auto dst_tensor = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    dst_tensor.Reshape({1, channels, height, width});

    dst = std::move(dst_tensor);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(HWC2CHW, (cpu, 0), []() { return std::make_unique<HWC2CHWImpl>(); });

}  // namespace mmdeploy::operation::cpu
