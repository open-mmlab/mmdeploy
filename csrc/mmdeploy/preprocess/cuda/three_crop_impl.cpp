// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/three_crop.h"

using namespace std;

namespace mmdeploy {
namespace cuda {

Result<Tensor> CropImage(Stream& stream, const Device& device, const Tensor& tensor, int top,
                         int left, int bottom, int right);

class ThreeCropImpl : public ::mmdeploy::ThreeCropImpl {
 public:
  explicit ThreeCropImpl(const Value& args) : ::mmdeploy::ThreeCropImpl(args) {}

 protected:
  Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                           int right) override {
    return ::mmdeploy::cuda::CropImage(stream_, device_, tensor, top, left, bottom, right);
  }
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::ThreeCropImpl, (cuda, 0), ThreeCropImpl);

}  // namespace cuda
}  // namespace mmdeploy
