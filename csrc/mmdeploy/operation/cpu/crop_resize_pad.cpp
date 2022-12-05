// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class CropResizePadImpl : public CropResizePad {
 public:
  CropResizePadImpl() = default;

  Result<void> apply(const Tensor &src, const std::vector<int> &crop_rect,
                     const std::vector<int> &target_size, const std::vector<int> &pad_rect,
                     Tensor &dst) override {
    auto src_mat = mmdeploy::cpu::Tensor2CVMat(src);
    auto dst_mat = mmdeploy::cpu::CropResizePad(src_mat, crop_rect, target_size, pad_rect);
    dst = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(CropResizePad, (cpu, 0),
                               []() { return std::make_unique<CropResizePadImpl>(); });

}  // namespace mmdeploy::operation::cpu
