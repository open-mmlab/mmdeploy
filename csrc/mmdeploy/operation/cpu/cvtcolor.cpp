// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class CvtColorImpl : public CvtColor {
 public:
  Result<void> apply(const Mat& src, Mat& dst, PixelFormat dst_fmt) override {
    auto src_mat = mmdeploy::cpu::Mat2CVMat(src);
    auto dst_mat = mmdeploy::cpu::CvtColor(src_mat, src.pixel_format(), dst_fmt);
    dst = mmdeploy::cpu::CVMat2Mat(dst_mat, dst_fmt);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(CvtColor, (cpu, 0), [] { return std::make_unique<CvtColorImpl>(); });

}  // namespace mmdeploy::operation::cpu
