// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/normalize.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy::cpu {

class NormalizeImpl : public ::mmdeploy::NormalizeImpl {
 public:
  NormalizeImpl(const Value& value) : ::mmdeploy::NormalizeImpl(value){};
  ~NormalizeImpl() = default;

 protected:
  Result<Tensor> NormalizeImage(const Tensor& tensor) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    auto mat = Tensor2CVMat(src_tensor);
    auto dst_mat = Normalize(mat, arg_.mean, arg_.std, arg_.to_rgb, true);
    return CVMat2Tensor(dst_mat);
  }

  Result<Tensor> ConvertToRGB(const Tensor& tensor) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));
    SyncOnScopeExit(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);
    auto src_mat = Tensor2CVMat(tensor);
    auto dst_mat = ColorTransfer(src_mat, PixelFormat::kBGR, PixelFormat::kRGB);
    return CVMat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::NormalizeImpl, (cpu, 0), NormalizeImpl);

}  // namespace mmdeploy::cpu
