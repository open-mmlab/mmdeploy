// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ToBGRImpl : public ToBGR {
 public:
  Result<void> apply(const Mat& src, Tensor& dst) override {
    auto src_mat = mmdeploy::cpu::Mat2CVMat(src);
    auto dst_mat = mmdeploy::cpu::ColorTransfer(src_mat, src.pixel_format(), PixelFormat::kBGR);
    dst = ::mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToBGR, (cpu, 0), []() { return std::make_unique<ToBGRImpl>(); });

class ToGrayImpl : public ToGray {
 public:
  Result<void> apply(const Mat& src, Tensor& dst) override {
    auto src_mat = mmdeploy::cpu::Mat2CVMat(src);
    auto dst_mat =
        mmdeploy::cpu::ColorTransfer(src_mat, src.pixel_format(), PixelFormat::kGRAYSCALE);
    dst = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToGray, (cpu, 0), []() { return std::make_unique<ToGrayImpl>(); });

}  // namespace mmdeploy::operation::cpu
