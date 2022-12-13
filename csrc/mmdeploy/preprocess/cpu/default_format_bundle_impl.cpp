// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/default_format_bundle.h"
#include "opencv_utils.h"

namespace mmdeploy::cpu {

class DefaultFormatBundleImpl : public ::mmdeploy::DefaultFormatBundleImpl {
 public:
  explicit DefaultFormatBundleImpl(const Value& args) : ::mmdeploy::DefaultFormatBundleImpl(args) {}

 protected:
  Result<Tensor> ToFloat32(const Tensor& tensor, const bool& img_to_float) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    auto data_type = src_tensor.desc().data_type;

    if (img_to_float && data_type == DataType::kINT8) {
      auto cvmat = Tensor2CVMat(src_tensor);
      cvmat.convertTo(cvmat, CV_32FC(cvmat.channels()));
      auto dst_tensor = CVMat2Tensor(cvmat);
      return dst_tensor;
    }
    return src_tensor;
  }

  Result<Tensor> HWC2CHW(const Tensor& tensor) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    auto shape = src_tensor.shape();
    int height = shape[1];
    int width = shape[2];
    int channels = shape[3];

    auto dst_mat = Transpose(Tensor2CVMat(src_tensor));

    auto dst_tensor = CVMat2Tensor(dst_mat);
    dst_tensor.Reshape({1, channels, height, width});

    return dst_tensor;
  }
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::DefaultFormatBundleImpl, (cpu, 0),
                                 DefaultFormatBundleImpl);

}  // namespace mmdeploy::cpu
