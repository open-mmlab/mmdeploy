// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class CropImpl : public Crop {
 public:
  using Crop::Crop;

  Result<Tensor> apply(const Tensor& img, int top, int left, int bottom, int right) override {
    cv::Mat mat = mmdeploy::cpu::Tensor2CVMat(img);
    cv::Mat cropped_mat = mmdeploy::cpu::Crop(mat, top, left, bottom, right);
    return mmdeploy::cpu::CVMat2Tensor(cropped_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Crop, (cpu, 0), [](const Context& context) {
  return std::make_unique<CropImpl>(context);
});

}  // namespace mmdeploy::operation::cpu
