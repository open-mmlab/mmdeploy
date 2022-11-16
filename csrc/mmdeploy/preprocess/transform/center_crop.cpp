// Copyright (c) OpenMMLab. All rights reserved.

#include <array>

#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::transform {

class CenterCrop : public Transform {
 public:
  explicit CenterCrop(const Value& args) {
    if (!args.contains(("crop_size"))) {
      throw std::invalid_argument("'crop_size' is expected");
    }
    if (args["crop_size"].is_number_integer()) {
      int crop_size = args["crop_size"].get<int>();
      crop_size_[0] = crop_size_[1] = crop_size;
    } else if (args["crop_size"].is_array() && args["crop_size"].size() == 2) {
      crop_size_[0] = args["crop_size"][0].get<int>();
      crop_size_[1] = args["crop_size"][1].get<int>();
    } else {
      throw std::invalid_argument("'crop_size' should be integer or an int array of size 2");
    }

    crop_ = operation::Managed<operation::Crop>::Create();
  }

  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
    auto img_fields = GetImageFields(input);

    for (auto& key : img_fields) {
      auto tensor = input[key].get<Tensor>();
      auto desc = tensor.desc();
      int h = desc.shape[1];
      int w = desc.shape[2];
      int crop_height = crop_size_[0];
      int crop_width = crop_size_[1];

      int y1 = std::max(0, int(std::round((h - crop_height) / 2.0)));
      int x1 = std::max(0, int(std::round((w - crop_width) / 2.0)));
      int y2 = std::min(h, y1 + crop_height) - 1;
      int x2 = std::min(w, x1 + crop_width) - 1;

      Tensor dst_tensor;
      OUTCOME_TRY(crop_.Apply(tensor, dst_tensor, y1, x1, y2, x2));

      auto& shape = dst_tensor.desc().shape;

      // trace static info & runtime args
      if (input.contains("__tracer__")) {
        input["__tracer__"].get_ref<Tracer&>().CenterCrop(
            {y1, x1, h - (int)shape[1] - y1, w - (int)shape[2] - x1},
            {(int)shape[1], (int)shape[2]}, tensor.data_type());
      }

      input["img_shape"] = {shape[0], shape[1], shape[2], shape[3]};
      if (input.contains("scale_factor")) {
        // image has been processed by `Resize` transform before.
        // Compute cropped image's offset against the original image
        assert(input["scale_factor"].is_array() && input["scale_factor"].size() >= 2);
        float w_scale = input["scale_factor"][0].get<float>();
        float h_scale = input["scale_factor"][1].get<float>();
        input["offset"].push_back(x1 / w_scale);
        input["offset"].push_back(y1 / h_scale);
      } else {
        input["offset"].push_back(x1);
        input["offset"].push_back(y1);
      }

      input[key] = std::move(dst_tensor);
    }

    MMDEPLOY_DEBUG("output: {}", to_json(input).dump(2));
    return success();
  }

 private:
  operation::Managed<operation::Crop> crop_;
  std::array<int, 2> crop_size_;
};

MMDEPLOY_REGISTER_TRANSFORM(CenterCrop);

}  // namespace mmdeploy::transform
