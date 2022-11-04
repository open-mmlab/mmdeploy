// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class CropImpl : public Crop {
 public:
  using Crop::Crop;

  Result<Tensor> crop(const Tensor& tensor, int top, int left, int bottom, int right) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device(), stream()));

    SyncOnScopeExit(stream(), src_tensor.buffer() != tensor.buffer(), src_tensor);

    cv::Mat mat = mmdeploy::cpu::Tensor2CVMat(src_tensor);
    cv::Mat cropped_mat = mmdeploy::cpu::Crop(mat, top, left, bottom, right);
    return mmdeploy::cpu::CVMat2Tensor(cropped_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Crop, (cpu, 0), [](const Context& context) {
  return std::make_unique<CropImpl>(context);
});

}  // namespace mmdeploy::operation::cpu
