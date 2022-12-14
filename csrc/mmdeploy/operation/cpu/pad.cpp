// Copyright (c) OpenMMLab. All rights reserved.

#include <map>

#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class PadImpl : public Pad {
 public:
  PadImpl(cv::BorderTypes border_type, float pad_val)
      : border_type_(border_type), pad_val_(pad_val) {}

  Result<void> apply(const Tensor& src, Tensor& dst, int top, int left, int bottom,
                     int right) override {
    cv::Mat dst_mat = mmdeploy::cpu::Pad(mmdeploy::cpu::Tensor2CVMat(src), top, left, bottom, right,
                                         border_type_, pad_val_);
    dst = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return success();
  }

 private:
  cv::BorderTypes border_type_;
  float pad_val_;
};

static auto Create(const string_view& border_type, float pad_val) {
  static const std::map<string_view, cv::BorderTypes> border_map{
      {"constant", cv::BORDER_CONSTANT},
      {"edge", cv::BORDER_REPLICATE},
      {"reflect", cv::BORDER_REFLECT_101},
      {"symmetric", cv::BORDER_REFLECT}};
  return std::make_unique<PadImpl>(border_map.at(border_type), pad_val);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Pad, (cpu, 0), Create);

}  // namespace mmdeploy::operation::cpu
