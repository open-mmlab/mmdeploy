// Copyright (c) OpenMMLab. All rights reserved.

#include "default_format_bundle.h"

#include <cassert>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/transform/tracer.h"

namespace mmdeploy {

DefaultFormatBundleImpl::DefaultFormatBundleImpl(const Value& args) : TransformImpl(args) {
  if (args.contains("img_to_float") && args["img_to_float"].is_boolean()) {
    arg_.img_to_float = args["img_to_float"].get<bool>();
  }
}

Result<Value> DefaultFormatBundleImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("DefaultFormatBundle input: {}", to_json(input).dump(2));
  Value output = input;
  if (input.contains("img")) {
    Tensor in_tensor = input["img"].get<Tensor>();
    OUTCOME_TRY(auto tensor, ToFloat32(in_tensor, arg_.img_to_float));

    // set default meta keys
    if (!output.contains("pad_shape")) {
      for (auto v : tensor.shape()) {
        output["pad_shape"].push_back(v);
      }
    }
    if (!output.contains("scale_factor")) {
      output["scale_factor"].push_back(1.0);
    }
    if (!output.contains("img_norm_cfg")) {
      int channel = tensor.shape()[3];
      for (int i = 0; i < channel; i++) {
        output["img_norm_cfg"]["mean"].push_back(0.0);
        output["img_norm_cfg"]["std"].push_back(1.0);
      }
      output["img_norm_cfg"]["to_rgb"] = false;
    }

    // trace static info & runtime args
    if (output.contains("__tracer__")) {
      output["__tracer__"].get_ref<Tracer&>().DefaultFormatBundle(arg_.img_to_float,
                                                                  in_tensor.data_type());
    }

    // transpose
    OUTCOME_TRY(tensor, HWC2CHW(tensor));
    SetTransformData(output, "img", std::move(tensor));
  }

  MMDEPLOY_DEBUG("DefaultFormatBundle output: {}", to_json(output).dump(2));
  return output;
}

DefaultFormatBundle::DefaultFormatBundle(const Value& args, int version) : Transform(args) {
  auto impl_creator = gRegistry<DefaultFormatBundleImpl>().Get(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'DefaultFormatBundle' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'DefaultFormatBundle' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (DefaultFormatBundle, 0), [](const Value& config) {
  return std::make_unique<DefaultFormatBundle>(config, 0);
});

MMDEPLOY_DEFINE_REGISTRY(DefaultFormatBundleImpl);
}  // namespace mmdeploy
