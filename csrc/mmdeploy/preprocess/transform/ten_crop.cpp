// Copyright (c) OpenMMLab. All rights reserved.

#include <array>

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::transform {

class TenCrop : public Transform {
 public:
  explicit TenCrop(const Value& args);
  ~TenCrop() override = default;

  Result<void> Apply(Value& data) override;

 protected:
  std::array<int, 2> crop_size_{};
  operation::Managed<operation::Crop> crop_;
  operation::Managed<operation::Flip> flip_;
};

TenCrop::TenCrop(const Value& args) {
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
  // horizontal flip
  flip_ = operation::Managed<operation::Flip>::Create(1);
}

Result<void> TenCrop::Apply(Value& data) {
  MMDEPLOY_DEBUG("input: {}", data);

  // copy input data, and update its properties
  Value output = data;
  auto tensor = data["img"].get<Tensor>();
  int img_h = tensor.shape(1);
  int img_w = tensor.shape(2);
  int crop_w = crop_size_[0];
  int crop_h = crop_size_[1];

  int w_step = (img_w - crop_w) / 4;
  int h_step = (img_h - crop_h) / 4;
  std::array<std::pair<int, int>, 5> offsets = {{{0, 0},
                                                 {4 * w_step, 0},
                                                 {0, 4 * h_step},
                                                 {4 * w_step, 4 * h_step},
                                                 {2 * w_step, 2 * h_step}}};
  vector<Tensor> cropped;
  cropped.reserve(10);
  for (const auto& [offx, offy] : offsets) {
    int y1 = offy;
    int y2 = offy + crop_h - 1;
    int x1 = offx;
    int x2 = offx + crop_w - 1;
    // ! No reallocation
    auto& cropped_tensor = cropped.emplace_back();
    auto& flipped_tensor = cropped.emplace_back();

    OUTCOME_TRY(crop_.Apply(tensor, cropped_tensor, y1, x1, y2, x2));
    OUTCOME_TRY(flip_.Apply(cropped_tensor, flipped_tensor));
  }

  Value::Array imgs;
  std::move(cropped.begin(), cropped.end(), std::back_inserter(imgs));
  data["imgs"] = std::move(imgs);

  return success();
}

MMDEPLOY_REGISTER_TRANSFORM(TenCrop);

}  // namespace mmdeploy::transform
