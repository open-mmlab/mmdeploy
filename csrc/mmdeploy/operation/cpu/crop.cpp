// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class CropImpl : public Crop {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst, int top, int left, int bottom,
                     int right) override {
    cv::Mat mat = mmdeploy::cpu::Tensor2CVMat(src);
    cv::Mat cropped_mat = mmdeploy::cpu::Crop(mat, top, left, bottom, right);
    dst = mmdeploy::cpu::CVMat2Tensor(cropped_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Crop, (cpu, 0), []() { return std::make_unique<CropImpl>(); });

}  // namespace mmdeploy::operation::cpu
