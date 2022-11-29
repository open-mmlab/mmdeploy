// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class WarpAffineImpl : public WarpAffine {
 public:
  explicit WarpAffineImpl(int method) : method_(method) {}

  Result<void> apply(const Tensor& src, Tensor& dst, const float affine_matrix[6], int dst_h,
                     int dst_w) override {
    auto src_mat = mmdeploy::cpu::Tensor2CVMat(src);
    cv::Mat_<float> _matrix(2, 3, const_cast<float*>(affine_matrix));
    auto dst_mat = mmdeploy::cpu::WarpAffine(src_mat, _matrix, dst_h, dst_w, method_);
    dst = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return success();
  }

 private:
  int method_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(WarpAffine, (cpu, 0), [](const string_view& interp) {
  return std::make_unique<WarpAffineImpl>(::mmdeploy::cpu::GetInterpolationMethod(interp).value());
});

}  // namespace mmdeploy::operation::cpu
