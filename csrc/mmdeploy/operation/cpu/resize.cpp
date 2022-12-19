// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ResizeImpl : public Resize {
 public:
  ResizeImpl(std::string interp) : interp_(std::move(interp)) {}

  Result<void> apply(const Tensor& src, Tensor& dst, int dst_h, int dst_w) override {
    auto src_mat = mmdeploy::cpu::Tensor2CVMat(src);
    auto dst_mat = mmdeploy::cpu::Resize(src_mat, dst_h, dst_w, interp_);
    dst = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return success();
  }

 private:
  std::string interp_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Resize, (cpu, 0), [](const string_view& interp) {
  return std::make_unique<ResizeImpl>(std::string(interp));
});

}  // namespace mmdeploy::operation::cpu
