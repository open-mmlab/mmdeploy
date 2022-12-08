// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class FlipImpl : public Flip {
 public:
  using Flip::Flip;

  Result<void> apply(const Tensor& src, Tensor& dst) override {
    cv::Mat mat = mmdeploy::cpu::Tensor2CVMat(src);
    cv::Mat flipped_mat;
    cv::flip(mat, flipped_mat, flip_code_);
    dst = mmdeploy::cpu::CVMat2Tensor(flipped_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Flip, (cpu, 0),
                               [](int flip_code) { return std::make_unique<FlipImpl>(flip_code); });

}  // namespace mmdeploy::operation::cpu
