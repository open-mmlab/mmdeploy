// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ToBGRImpl : public ToBGR {
 public:
  using ToBGR::ToBGR;

  Result<Tensor> apply(const Mat& img) override {
    auto src_mat = mmdeploy::cpu::Mat2CVMat(img);
    auto dst_mat = mmdeploy::cpu::ColorTransfer(src_mat, img.pixel_format(), PixelFormat::kBGR);
    return ::mmdeploy::cpu::CVMat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToBGR, (cpu, 0), [](const Context& context) {
  return std::make_unique<ToBGRImpl>(context);
});

class ToGrayImpl : public ToGray {
 public:
  using ToGray::ToGray;

  Result<Tensor> apply(const Mat& img) override {
    auto src_mat = mmdeploy::cpu::Mat2CVMat(img);
    auto dst_mat =
        mmdeploy::cpu::ColorTransfer(src_mat, img.pixel_format(), PixelFormat::kGRAYSCALE);

    return ::mmdeploy::cpu::CVMat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToGray, (cpu, 0), [](const Context& context) {
  return std::make_unique<ToGrayImpl>(context);
});

}  // namespace mmdeploy::operation::cpu
