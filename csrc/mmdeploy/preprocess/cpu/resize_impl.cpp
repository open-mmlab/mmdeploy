// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/resize.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy::cpu {

class ResizeImpl final : public ::mmdeploy::ResizeImpl {
 public:
  ResizeImpl(const Value& args) : ::mmdeploy::ResizeImpl(args) {}
  ~ResizeImpl() = default;

 protected:
  Result<Tensor> ResizeImage(const Tensor& img, int dst_h, int dst_w) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device_, stream_));

    SyncOnScopeExit(stream_, src_tensor.buffer() != img.buffer(), src_tensor);

    auto src_mat = Tensor2CVMat(src_tensor);
    auto dst_mat = Resize(src_mat, dst_h, dst_w, arg_.interpolation);

    return CVMat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::ResizeImpl, (cpu, 0), ResizeImpl);

}  // namespace mmdeploy::cpu
