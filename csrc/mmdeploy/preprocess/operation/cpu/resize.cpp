// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ResizeImpl : public Resize {
 public:
  using Resize::Resize;

  Result<Tensor> resize(const Tensor& img, int dst_h, int dst_w) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device(), stream()));
    SyncOnScopeExit(stream(), src_tensor.buffer() != img.buffer(), src_tensor);

    auto src_mat = mmdeploy::cpu::Tensor2CVMat(src_tensor);
    auto dst_mat = mmdeploy::cpu::Resize(src_mat, dst_h, dst_w, interp_);

    return mmdeploy::cpu::CVMat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Resize, (cpu, 0),
                               [](const string_view& interp, const Context& context) {
                                 return std::make_unique<ResizeImpl>(interp, context);
                               });

}  // namespace mmdeploy::operation::cpu
