// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/crop.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy::cpu {

class CenterCropImpl : public ::mmdeploy::CenterCropImpl {
 public:
  explicit CenterCropImpl(const Value& args) : ::mmdeploy::CenterCropImpl(args) {}

 protected:
  Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                           int right) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    cv::Mat mat = Tensor2CVMat(src_tensor);
    cv::Mat cropped_mat = Crop(mat, top, left, bottom, right);
    return CVMat2Tensor(cropped_mat);
  }
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::CenterCropImpl, (cpu, 0), CenterCropImpl);

}  // namespace mmdeploy::cpu
