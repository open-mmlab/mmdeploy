// Copyright (c) OpenMMLab. All rights reserved.

#include <array>

#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::transform {

namespace {

Result<void> check_input_shape(int img_h, int img_w, int crop_h, int crop_w) {
  if (img_h == crop_h || img_w == crop_w) {
    return success();
  }
  MMDEPLOY_ERROR("ThreeCrop error, img_h: {} != crop_h: {} && img_w: {} != crop_w {}", img_h,
                 crop_h, img_w, crop_w);
  return Status(eInvalidArgument);
}

}  // namespace

class ThreeCrop : public Transform {
 public:
  explicit ThreeCrop(const Value& args);
  ~ThreeCrop() override = default;

  Result<void> Apply(Value& data) override;

 protected:
  std::array<int, 2> crop_size_{};
  operation::Managed<operation::Crop> crop_;
};

ThreeCrop::ThreeCrop(const Value& args) {
  // (w, h) of crop size
  if (!args.contains(("crop_size"))) {
    MMDEPLOY_ERROR("'crop_size' is expected");
    throw_exception(eInvalidArgument);
  }
  if (args["crop_size"].is_number_integer()) {
    crop_size_[0] = crop_size_[1] = args["crop_size"].get<int>();
  } else if (args["crop_size"].is_array() && args["crop_size"].size() == 2) {
    crop_size_[0] = args["crop_size"][0].get<int>();
    crop_size_[1] = args["crop_size"][1].get<int>();
  } else {
    MMDEPLOY_ERROR("'crop_size' should be integer or an int array of size 2");
    throw_exception(eInvalidArgument);
  }

  crop_ = operation::Managed<operation::Crop>::Create();
}

Result<void> ThreeCrop::Apply(Value& data) {
  auto tensor = data["img"].get<Tensor>();
  auto desc = tensor.desc();
  int img_h = desc.shape[1];
  int img_w = desc.shape[2];
  int crop_w = crop_size_[0];
  int crop_h = crop_size_[1];
  OUTCOME_TRY(check_input_shape(img_h, img_w, crop_h, crop_w));

  std::array<std::pair<int, int>, 3> offsets;
  if (crop_h == img_h) {
    int w_step = (img_w - crop_w) / 2;
    offsets = {{{0, 0}, {2 * w_step, 0}, {w_step, 0}}};
  } else if (crop_w == img_w) {
    int h_step = (img_h - crop_h) / 2;
    offsets = {{{0, 0}, {0, 2 * h_step}, {0, h_step}}};
  }
  vector<Tensor> cropped;
  cropped.reserve(3);
  for (const auto& [offx, offy] : offsets) {
    int y1 = offy;
    int y2 = offy + crop_h - 1;
    int x1 = offx;
    int x2 = offx + crop_w - 1;
    auto& dst_tensor = cropped.emplace_back();

    OUTCOME_TRY(crop_.Apply(tensor, dst_tensor, y1, x1, y2, x2));
  }

  Value::Array imgs;
  std::move(cropped.begin(), cropped.end(), std::back_inserter(imgs));
  data["imgs"] = std::move(imgs);
  return success();
}

MMDEPLOY_REGISTER_TRANSFORM(ThreeCrop);

}  // namespace mmdeploy::transform
