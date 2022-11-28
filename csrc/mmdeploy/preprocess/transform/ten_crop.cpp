// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/ten_crop.h"

#include "mmdeploy/archive/json_archive.h"

using namespace std;

namespace mmdeploy {

TenCropImpl::TenCropImpl(const Value& args) : TransformImpl(args) {
  // (w, h) of crop size
  if (!args.contains(("crop_size"))) {
    throw std::invalid_argument("'crop_size' is expected");
  }
  if (args["crop_size"].is_number_integer()) {
    int crop_size = args["crop_size"].get<int>();
    arg_.crop_size[0] = arg_.crop_size[1] = crop_size;
  } else if (args["crop_size"].is_array() && args["crop_size"].size() == 2) {
    arg_.crop_size[0] = args["crop_size"][0].get<int>();
    arg_.crop_size[1] = args["crop_size"][1].get<int>();
  } else {
    throw std::invalid_argument("'crop_size' should be integer or an int array of size 2");
  }
}

Result<Value> TenCropImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));

  // copy input data, and update its properties
  Value output = input;
  auto tensor = input["img"].get<Tensor>();
  int img_h = tensor.shape(1);
  int img_w = tensor.shape(2);
  int crop_w = arg_.crop_size[0];
  int crop_h = arg_.crop_size[1];

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
    OUTCOME_TRY(auto cropped_tensor, CropImage(tensor, y1, x1, y2, x2));
    OUTCOME_TRY(auto flipped_tensor, HorizontalFlip(cropped_tensor));
    cropped.push_back(std::move(cropped_tensor));
    cropped.push_back(std::move(flipped_tensor));
  }

  output["imgs"] = Value{};
  for (int i = 0; i < cropped.size(); i++) {
    output["imgs"].push_back(cropped[i]);
    output["__data__"].push_back(std::move(cropped[i]));
  }

  return output;
}

TenCrop::TenCrop(const Value& args, int version) : Transform(args) {
  auto impl_creator = gRegistry<TenCropImpl>().Get(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'TenCrop' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Resize' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (TenCrop, 0), [](const Value& config) {
  return std::make_unique<TenCrop>(config, 0);
});

MMDEPLOY_DEFINE_REGISTRY(TenCropImpl);
}  // namespace mmdeploy
