// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/image2tensor.h"
#include "opencv_utils.h"

namespace mmdeploy {
namespace cpu {

class ImageToTensorImpl : public ::mmdeploy::ImageToTensorImpl {
 public:
  explicit ImageToTensorImpl(const Value& args) : ::mmdeploy::ImageToTensorImpl(args) {}

 protected:
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

class ImageToTensorImplCreator : public Creator<::mmdeploy::ImageToTensorImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override {
    return std::make_unique<ImageToTensorImpl>(args);
  }
};

}  // namespace cpu
}  // namespace mmdeploy

using mmdeploy::ImageToTensorImpl;
using mmdeploy::cpu::ImageToTensorImplCreator;
REGISTER_MODULE(ImageToTensorImpl, ImageToTensorImplCreator);
