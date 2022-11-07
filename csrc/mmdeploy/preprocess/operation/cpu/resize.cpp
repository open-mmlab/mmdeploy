// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ResizeImpl : public Resize {
 public:
  using Resize::Resize;

  Result<Tensor> apply(const Tensor& img, int dst_h, int dst_w) override {
    auto src_mat = mmdeploy::cpu::Tensor2CVMat(img);
    auto dst_mat = mmdeploy::cpu::Resize(src_mat, dst_h, dst_w, interp_);
    return mmdeploy::cpu::CVMat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Resize, (cpu, 0),
                               [](const string_view& interp, const Context& context) {
                                 return std::make_unique<ResizeImpl>(interp, context);
                               });

}  // namespace mmdeploy::operation::cpu
