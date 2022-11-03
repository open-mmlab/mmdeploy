// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/load.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy::cpu {

class PrepareImageImpl : public ::mmdeploy::PrepareImageImpl {
 public:
  explicit PrepareImageImpl(const Value& args) : ::mmdeploy::PrepareImageImpl(args){};
  ~PrepareImageImpl() override = default;

 protected:
  Result<Tensor> ConvertToBGR(const Mat& img) override {
    auto src_mat = Mat2CVMat(img);
    auto dst_mat = ColorTransfer(src_mat, img.pixel_format(), PixelFormat::kBGR);
    if (arg_.to_float32) {
      cv::Mat _dst_mat;
      dst_mat.convertTo(_dst_mat, CV_32FC3);
      dst_mat = _dst_mat;
    }
    return ::mmdeploy::cpu::CVMat2Tensor(dst_mat);
  }

  Result<Tensor> ConvertToGray(const Mat& img) override {
    auto src_mat = Mat2CVMat(img);
    auto dst_mat = ColorTransfer(src_mat, img.pixel_format(), PixelFormat::kGRAYSCALE);
    if (arg_.to_float32) {
      cv::Mat _dst_mat;
      dst_mat.convertTo(_dst_mat, CV_32FC1);
      dst_mat = _dst_mat;
    }
    return ::mmdeploy::cpu::CVMat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::PrepareImageImpl, (cpu, 0), PrepareImageImpl);

}  // namespace mmdeploy::cpu
