// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/three_crop.h"

#include "mmdeploy/archive/json_archive.h"

using namespace std;

namespace mmdeploy {

Result<void> check_input_shape(int img_h, int img_w, int crop_h, int crop_w) {
  if (img_h == crop_h || img_w == crop_w) {
    return success();
  }
  MMDEPLOY_ERROR("ThreeCrop error, img_h: {} != crop_h: {} && img_w: {} != crop_w {}", img_h,
                 crop_h, img_w, crop_w);
  return Status(eInvalidArgument);
}

ThreeCropImpl::ThreeCropImpl(const Value& args) : TransformImpl(args) {
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

Result<Value> ThreeCropImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));

  // copy input data, and update its properties
  Value output = input;
  auto tensor = input["img"].get<Tensor>();
  auto desc = tensor.desc();
  int img_h = desc.shape[1];
  int img_w = desc.shape[2];
  int crop_w = arg_.crop_size[0];
  int crop_h = arg_.crop_size[1];
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
    OUTCOME_TRY(auto dst_tensor, CropImage(tensor, y1, x1, y2, x2));
    cropped.push_back(std::move(dst_tensor));
  }

  output["imgs"] = Value{};
  for (int i = 0; i < cropped.size(); i++) {
    output["imgs"].push_back(cropped[i]);
    output["__data__"].push_back(std::move(cropped[i]));
  }

  return output;
}

ThreeCrop::ThreeCrop(const Value& args, int version) : Transform(args) {
  auto impl_creator = gRegistry<ThreeCropImpl>().Get(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'ThreeCrop' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Resize' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (ThreeCrop, 0), [](const Value& config) {
  return std::make_unique<ThreeCrop>(config, 0);
});

MMDEPLOY_DEFINE_REGISTRY(ThreeCropImpl);

}  // namespace mmdeploy
