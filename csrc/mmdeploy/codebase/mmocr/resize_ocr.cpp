// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::mmocr {

class ResizeOCR : public transform::Transform {
 public:
  explicit ResizeOCR(const Value& args) noexcept {
    height_ = args.value("height", height_);
    min_width_ = args.contains("min_width") && args["min_width"].is_number_integer()
                     ? args["min_width"].get<int>()
                     : min_width_;
    max_width_ = args.contains("max_width") && args["max_width"].is_number_integer()
                     ? args["max_width"].get<int>()
                     : max_width_;
    keep_aspect_ratio_ = args.value("keep_aspect_ratio", keep_aspect_ratio_);
    backend_ = args.contains("backend") && args["backend"].is_string()
                   ? args["backend"].get<string>()
                   : backend_;
    img_pad_value_ = args.value("img_pad_value", img_pad_value_);
    width_downsample_ratio_ = args.value("width_downsample_ratio", width_downsample_ratio_);

    resize_ = operation::Managed<operation::Resize>::Create("bilinear");
    pad_ = operation::Managed<operation::Pad>::Create("constant", img_pad_value_);
  }

  ~ResizeOCR() override = default;

  Result<void> Apply(Value& data) override {
    MMDEPLOY_DEBUG("input: {}", data);
    auto dst_height = height_;
    auto dst_min_width = min_width_;
    auto dst_max_width = max_width_;

    std::vector<int> img_shape;  // NHWC
    from_value(data["img_shape"], img_shape);

    std::vector<int> ori_shape;  // NHWC
    from_value(data["ori_shape"], ori_shape);

    auto ori_height = ori_shape[1];
    auto ori_width = ori_shape[2];
    auto valid_ratio = 1.f;

    auto img = data["img"].get<Tensor>();
    Tensor img_resize;
    if (keep_aspect_ratio_) {
      auto new_width = static_cast<int>(std::ceil(1.f * dst_height / ori_height * ori_width));
      auto width_divisor = static_cast<int>(1 / width_downsample_ratio_);
      if (new_width % width_divisor != 0) {
        new_width = std::round(1.f * new_width / width_divisor) * width_divisor;
      }
      if (dst_min_width > 0) {
        new_width = std::max(dst_min_width, new_width);
      }
      if (dst_max_width > 0) {
        valid_ratio = std::min(1., 1. * new_width / dst_max_width);
        auto resize_width = std::min(dst_max_width, new_width);
        OUTCOME_TRY(resize_.Apply(img, img_resize, dst_height, resize_width));
        if (new_width < dst_max_width) {
          auto pad_w = std::max(0, dst_max_width - resize_width);
          OUTCOME_TRY(pad_.Apply(img_resize, img_resize, 0, 0, 0, pad_w));
        }
      } else {
        OUTCOME_TRY(resize_.Apply(img, img_resize, dst_height, new_width));
      }
    } else {
      OUTCOME_TRY(resize_.Apply(img, img_resize, dst_height, dst_max_width));
    }

    data["img"] = img_resize;
    data["resize_shape"] = to_value(img_resize.desc().shape);
    data["pad_shape"] = data["resize_shape"];
    data["valid_ratio"] = valid_ratio;
    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

 protected:
  operation::Managed<operation::Resize> resize_;
  operation::Managed<operation::Pad> pad_;
  int height_{-1};
  int min_width_{-1};
  int max_width_{-1};
  bool keep_aspect_ratio_{true};
  float img_pad_value_{0};
  float width_downsample_ratio_{1. / 16};
  std::string backend_;
};

MMDEPLOY_REGISTER_TRANSFORM(ResizeOCR);

}  // namespace mmdeploy::mmocr
