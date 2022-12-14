// Copyright (c) OpenMMLab. All rights reserved.

#include <set>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::mmocr {

class RescaleToHeight : public transform::Transform {
 public:
  explicit RescaleToHeight(const Value& args) noexcept {
    height_ = args.value("height", height_);
    min_width_ = args.contains("min_width") && args["min_width"].is_number_integer()
                     ? args["min_width"].get<int>()
                     : min_width_;
    max_width_ = args.contains("max_width") && args["max_width"].is_number_integer()
                     ? args["max_width"].get<int>()
                     : max_width_;
    width_divisor_ = args.contains("width_divisor") && args["width_divisor"].is_number_integer()
                         ? args["width_divisor"].get<int>()
                         : width_divisor_;
    resize_ = operation::Managed<operation::Resize>::Create("bilinear");
  }

  ~RescaleToHeight() override = default;

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
    auto new_width = static_cast<int>(std::ceil(1.f * dst_height / ori_height * ori_width));
    auto width_divisor = width_divisor_;
    if (dst_min_width > 0) {
      new_width = std::max(dst_min_width, new_width);
    }
    if (dst_max_width > 0) {
      new_width = std::min(dst_max_width, new_width);
    }
    if (new_width % width_divisor != 0) {
      new_width = std::round(1.f * new_width / width_divisor) * width_divisor;
    }
    OUTCOME_TRY(resize_.Apply(img, img_resize, dst_height, new_width));
    data["img"] = img_resize;
    data["resize_shape"] = to_value(img_resize.desc().shape);
    data["pad_shape"] = data["resize_shape"];
    data["ori_shape"] = data["ori_shape"];
    data["scale"] = to_value(std::vector<int>({new_width, dst_height}));
    data["valid_ratio"] = valid_ratio;
    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

 protected:
  operation::Managed<operation::Resize> resize_;
  int height_{-1};
  int min_width_{-1};
  int max_width_{-1};
  bool keep_aspect_ratio_{true};
  int width_divisor_{1};
};

MMDEPLOY_REGISTER_TRANSFORM(RescaleToHeight);
}  // namespace mmdeploy::mmocr
