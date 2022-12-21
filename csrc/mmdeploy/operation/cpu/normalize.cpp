// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class NormalizeImpl : public Normalize {
 public:
  explicit NormalizeImpl(Param param) : param_(std::move(param)) {}

  Result<void> apply(const Tensor& src, Tensor& dst) override {
    auto mat = mmdeploy::cpu::Tensor2CVMat(src);
    auto dst_mat = mmdeploy::cpu::Normalize(mat, param_.mean, param_.std, param_.to_rgb, false);
    auto output = mmdeploy::cpu::CVMat2Tensor(dst_mat);

    dst = std::move(output);
    return success();
  }

 protected:
  Param param_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Normalize, (cpu, 0), [](const Normalize::Param& param) {
  return std::make_unique<NormalizeImpl>(param);
});

}  // namespace mmdeploy::operation::cpu
