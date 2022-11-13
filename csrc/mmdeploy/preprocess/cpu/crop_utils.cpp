// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/crop.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

Result<Tensor> CropImage(Stream& stream, const Device& device, const Tensor& tensor, int top,
                         int left, int bottom, int right) {
  OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device, stream));

  SyncOnScopeExit(stream, src_tensor.buffer() != tensor.buffer(), src_tensor);

  cv::Mat mat = Tensor2CVMat(src_tensor);
  cv::Mat cropped_mat = Crop(mat, top, left, bottom, right);
  return CVMat2Tensor(cropped_mat);
}

}  // namespace cpu
}  // namespace mmdeploy
