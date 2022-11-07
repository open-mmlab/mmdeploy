// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class NormalizeImpl : public Normalize {
 public:
  NormalizeImpl(Param param, const Context& context)
      : Normalize(context), param_(std::move(param)) {}

  Result<Tensor> apply(const Tensor& img) override {
    auto mat = mmdeploy::cpu::Tensor2CVMat(img);
    auto dst_mat = mmdeploy::cpu::Normalize(mat, param_.mean, param_.std, param_.to_rgb, true);
    auto output = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return output;
  }

 protected:
  Param param_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Normalize, (cpu, 0),
                               [](const Normalize::Param& param, const Context& context) {
                                 return std::make_unique<NormalizeImpl>(param, context);
                               });

}  // namespace mmdeploy::operation::cpu
