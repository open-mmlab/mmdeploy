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

class ShortScaleAspectJitter : public transform::Transform {
 public:
  explicit ShortScaleAspectJitter(const Value& args) noexcept {
    short_size_ = args.contains("short_size") && args["short_size"].is_number_integer()
                      ? args["short_size"].get<int>()
                      : short_size_;
    if (args["ratio_range"].is_array() && args["ratio_range"].size() == 2) {
      ratio_range_[0] = args["ratio_range"][0].get<float>();
      ratio_range_[1] = args["ratio_range"][1].get<float>();
    } else {
      MMDEPLOY_ERROR("'ratio_range' should be a float array of size 2");
      throw_exception(eInvalidArgument);
    }

    if (args["aspect_ratio_range"].is_array() && args["aspect_ratio_range"].size() == 2) {
      aspect_ratio_range_[0] = args["aspect_ratio_range"][0].get<float>();
      aspect_ratio_range_[1] = args["aspect_ratio_range"][1].get<float>();
    } else {
      MMDEPLOY_ERROR("'aspect_ratio_range' should be a float array of size 2");
      throw_exception(eInvalidArgument);
    }
    scale_divisor_ = args.contains("scale_divisor") && args["scale_divisor"].is_number_integer()
                         ? args["scale_divisor"].get<int>()
                         : scale_divisor_;
    resize_ = operation::Managed<operation::Resize>::Create("bilinear");
  }

  ~ShortScaleAspectJitter() override = default;

  Result<void> Apply(Value& data) override {
    MMDEPLOY_DEBUG("input: {}", data);
    auto short_size = short_size_;
    auto ratio_range = ratio_range_;
    auto aspect_ratio_range = aspect_ratio_range_;
    auto scale_divisor = scale_divisor_;

    if (ratio_range[0] != 1.0 || ratio_range[1] != 1.0 || aspect_ratio_range[0] != 1.0 ||
        aspect_ratio_range[1] != 1.0) {
      MMDEPLOY_ERROR("unsupported `ratio_range` and `aspect_ratio_range`");
      return Status(eNotSupported);
    }
    std::vector<int> img_shape;  // NHWC
    from_value(data["img_shape"], img_shape);

    std::vector<int> ori_shape;  // NHWC
    from_value(data["ori_shape"], ori_shape);

    auto ori_height = ori_shape[1];
    auto ori_width = ori_shape[2];

    auto img = data["img"].get<Tensor>();
    Tensor img_resize;
    auto scale = static_cast<float>(1.0 * short_size / std::min(img_shape[1], img_shape[2]));
    auto dst_height = static_cast<int>(std::round(scale * img_shape[1]));
    auto dst_width = static_cast<int>(std::round(scale * img_shape[2]));
    dst_height = static_cast<int>(std::ceil(1.0 * dst_height / scale_divisor) * scale_divisor);
    dst_width = static_cast<int>(std::ceil(1.0 * dst_width / scale_divisor) * scale_divisor);
    std::vector<float> scale_factor = {(float)1.0 * dst_width / img_shape[2],
                                       (float)1.0 * dst_height / img_shape[1]};

    OUTCOME_TRY(resize_.Apply(img, img_resize, dst_height, dst_width));
    data["img"] = img_resize;
    data["resize_shape"] = to_value(img_resize.desc().shape);
    data["scale"] = to_value(std::vector<int>({dst_width, dst_height}));
    data["scale_factor"] = to_value(scale_factor);
    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

 protected:
  operation::Managed<operation::Resize> resize_;
  int short_size_{736};
  std::vector<float> ratio_range_{0.7, 1.3};
  std::vector<float> aspect_ratio_range_{0.9, 1.1};
  int scale_divisor_{1};
};

MMDEPLOY_REGISTER_TRANSFORM(ShortScaleAspectJitter);

}  // namespace mmdeploy::mmocr
