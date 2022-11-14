// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/three_crop.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

Result<Tensor> CropImage(Stream& stream, const Device& device, const Tensor& tensor, int top,
                         int left, int bottom, int right);

class ThreeCropImpl : public ::mmdeploy::ThreeCropImpl {
 public:
  explicit ThreeCropImpl(const Value& args) : ::mmdeploy::ThreeCropImpl(args) {}

 protected:
  Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                           int right) override {
    return ::mmdeploy::cpu::CropImage(stream_, device_, tensor, top, left, bottom, right);
  }
};

class ThreeCropImplCreator : public Creator<::mmdeploy::ThreeCropImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<ThreeCropImpl>(args); }
};

}  // namespace cpu
}  // namespace mmdeploy

using ::mmdeploy::ThreeCropImpl;
using ::mmdeploy::cpu::ThreeCropImplCreator;
REGISTER_MODULE(ThreeCropImpl, ThreeCropImplCreator);
