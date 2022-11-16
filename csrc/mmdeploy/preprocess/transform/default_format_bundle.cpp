// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

class DefaultFormatBundle : public Transform {
 public:
  explicit DefaultFormatBundle(const Value& args) {
    if (args.contains("img_to_float") && args["img_to_float"].is_boolean()) {
      img_to_float_ = args["img_to_float"].get<bool>();
    }
    to_float_ = operation::Managed<operation::ToFloat>::Create();
    hwc2chw_ = operation::Managed<operation::HWC2CHW>::Create();
  }

  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("DefaultFormatBundle input: {}", to_json(input).dump(2));

    if (input.contains("img")) {
      Tensor tensor = input["img"].get<Tensor>();
      auto input_data_type = tensor.data_type();
      if (img_to_float_) {
        OUTCOME_TRY(to_float_.Apply(tensor, tensor));
      }

      // set default meta keys
      if (!input.contains("pad_shape")) {
        for (auto v : tensor.shape()) {
          input["pad_shape"].push_back(v);
        }
      }
      if (!input.contains("scale_factor")) {
        input["scale_factor"].push_back(1.0);
      }
      if (!input.contains("img_norm_cfg")) {
        int channel = tensor.shape()[3];
        for (int i = 0; i < channel; i++) {
          input["img_norm_cfg"]["mean"].push_back(0.0);
          input["img_norm_cfg"]["std"].push_back(1.0);
        }
        input["img_norm_cfg"]["to_rgb"] = false;
      }

      // trace static info & runtime args
      if (input.contains("__tracer__")) {
        input["__tracer__"].get_ref<Tracer&>().DefaultFormatBundle(img_to_float_, input_data_type);
      }

      // transpose
      OUTCOME_TRY(hwc2chw_.Apply(tensor, tensor));
      input["img"] = std::move(tensor);
    }

    MMDEPLOY_DEBUG("DefaultFormatBundle output: {}", to_json(input).dump(2));
    return success();
  }

 private:
  operation::Managed<operation::ToFloat> to_float_;
  operation::Managed<operation::HWC2CHW> hwc2chw_;
  bool img_to_float_ = true;
};

MMDEPLOY_REGISTER_TRANSFORM(DefaultFormatBundle);

}  // namespace mmdeploy::transform
