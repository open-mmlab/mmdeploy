// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

class DefaultFormatBundle : public Transform {
 public:
  explicit DefaultFormatBundle(const Value& args) {
    if (args.contains("img_to_float") && args["img_to_float"].is_boolean()) {
      img_to_float_ = args["img_to_float"].get<bool>();
    }
    auto context = GetContext(args);
    to_float_ = operation::Create<operation::ToFloat>(context.device, context);
    hwc2chw_ = operation::Create<operation::HWC2CHW>(context.device, context);
  }

  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("DefaultFormatBundle input: {}", to_json(input).dump(2));

    if (input.contains("img")) {
      Tensor tensor = input["img"].get<Tensor>();
      auto input_data_type = tensor.data_type();
      if (img_to_float_) {
        OUTCOME_TRY(tensor, to_float_->to_float(tensor));
        SetTransformData(input, "img", tensor);
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
      OUTCOME_TRY(tensor, hwc2chw_->hwc2chw(tensor));
      SetTransformData(input, "img", std::move(tensor));
    }

    MMDEPLOY_DEBUG("DefaultFormatBundle output: {}", to_json(input).dump(2));
    return success();
  }

 private:
  std::unique_ptr<operation::ToFloat> to_float_;
  std::unique_ptr<operation::HWC2CHW> hwc2chw_;
  bool img_to_float_ = true;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (DefaultFormatBundle, 0), [](const Value& config) {
  return std::make_unique<DefaultFormatBundle>(config);
});

}  // namespace mmdeploy::transform
