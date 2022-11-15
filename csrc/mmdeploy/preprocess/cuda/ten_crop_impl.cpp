// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/ten_crop.h"
#include "ppl/cv/cuda/flip.h"

using namespace std;

namespace mmdeploy {
namespace cuda {

Result<Tensor> CropImage(Stream& stream, const Device& device, const Tensor& tensor, int top,
                         int left, int bottom, int right);

class TenCropImpl : public ::mmdeploy::TenCropImpl {
 public:
  explicit TenCropImpl(const Value& args) : ::mmdeploy::TenCropImpl(args) {}

 protected:
  Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                           int right) override {
    return ::mmdeploy::cuda::CropImage(stream_, device_, tensor, top, left, bottom, right);
  }

  Result<Tensor> HorizontalFlip(const Tensor& tensor) {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit sync(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    TensorDesc dst_desc = tensor.desc();
    dst_desc.device = device_;
    Tensor dst_tensor(dst_desc);
    auto stream = GetNative<cudaStream_t>(stream_);
    int h = (int)tensor.shape(1);
    int w = (int)tensor.shape(2);
    int c = (int)tensor.shape(3);
    ppl::common::RetCode ret;
    if (tensor.data_type() == DataType::kINT8) {
      auto input = tensor.data<uint8_t>();
      auto output = dst_tensor.data<uint8_t>();
      if (c == 1) {
        ret = ppl::cv::cuda::Flip<uint8_t, 1>(stream, h, w, w * c, input, w * c, output, 1);
      } else if (c == 3) {
        ret = ppl::cv::cuda::Flip<uint8_t, 3>(stream, h, w, w * c, input, w * c, output, 1);
      } else {
        ret = ppl::common::RC_UNSUPPORTED;
      }
    } else if (tensor.data_type() == DataType::kFLOAT) {
      auto input = tensor.data<float>();
      auto output = dst_tensor.data<float>();
      if (c == 1) {
        ret = ppl::cv::cuda::Flip<float, 1>(stream, h, w, w * c, input, w * c, output, 1);
      } else if (c == 3) {
        ret = ppl::cv::cuda::Flip<float, 3>(stream, h, w, w * c, input, w * c, output, 1);
      } else {
        ret = ppl::common::RC_UNSUPPORTED;
      }
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", tensor.data_type());
      return Status(eNotSupported);
    }

    if (ret != 0) {
      return Status(eFail);
    }

    return dst_tensor;
  }
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::TenCropImpl, (cuda, 0), TenCropImpl);

}  // namespace cuda
}  // namespace mmdeploy
