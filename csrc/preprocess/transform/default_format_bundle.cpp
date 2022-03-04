// Copyright (c) OpenMMLab. All rights reserved.

#include "default_format_bundle.h"

#include <cassert>

#include "archive/json_archive.h"
#include "core/tensor.h"

namespace mmdeploy {

DefaultFormatBundleImpl::DefaultFormatBundleImpl(const Value& args) : TransformImpl(args) {
  if (args.contains("img_to_float") && args["img_to_float"].is_boolean()) {
    arg_.img_to_float = args["img_to_float"].get<bool>();
  }
  if (args.contains("pad_val") && args["pad_val"].is_object()) {
    for (auto& [key, _] : arg_.pad_val)
      if (args["pad_val"].contains(key) && args["pad_val"][key].is_number_float()) {
        arg_.pad_val[key] = args["pad_val"][key].get<float>();
      }
  }
}

Result<Value> DefaultFormatBundleImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("DefaultFormatBundle input: {}", to_json(input).dump(2));
  Value output = input;
  if (input.contains("img")) {
    Tensor tensor = input["img"].get<Tensor>();
    OUTCOME_TRY(output["img"], ToFloat32(tensor, arg_.img_to_float));
  }

  Tensor tensor = output["img"].get<Tensor>();
  // set default meta keys
  for (auto v : tensor.shape()) {
    output["pad_shape"].push_back(v);
  }
  output["scale_factor"].push_back(1.0);
  int channel = tensor.shape()[3];
  for (int i = 0; i < channel; i++) {
    output["img_norm_cfg"]["mean"].push_back(0.0);
    output["img_norm_cfg"]["std"].push_back(1.0);
  }
  output["img_norm_cfg"]["to_rgb"] = false;

  // transpose
  OUTCOME_TRY(output["img"], HWC2CHW(tensor));

  MMDEPLOY_DEBUG("DefaultFormatBundle output: {}", to_json(output).dump(2));
  return output;
}

DefaultFormatBundle::DefaultFormatBundle(const Value& args, int version) : Transform(args) {
  auto impl_creator =
      Registry<DefaultFormatBundleImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'DefaultFormatBundle' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'DefaultFormatBundle' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class DefaultFormatBundleCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "DefaultFormatBundle"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override {
    return std::make_unique<DefaultFormatBundle>(args, version_);
  }

 private:
  int version_{1};
};
REGISTER_MODULE(Transform, DefaultFormatBundleCreator);
MMDEPLOY_DEFINE_REGISTRY(DefaultFormatBundleImpl);
}  // namespace mmdeploy
