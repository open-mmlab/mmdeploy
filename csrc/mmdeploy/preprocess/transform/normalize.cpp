// Copyright (c) OpenMMLab. All rights reserved.

#include "normalize.h"

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/tracer.h"

using namespace std;

namespace mmdeploy {

// MMDEPLOY_DEFINE_REGISTRY(NormalizeImpl);

NormalizeImpl::NormalizeImpl(const Value& args) : TransformImpl(args) {
  if (!args.contains("mean") || !args.contains("std")) {
    MMDEPLOY_ERROR("no 'mean' or 'std' is configured");
    throw std::invalid_argument("no 'mean' or 'std' is configured");
  }
  for (auto& v : args["mean"]) {
    arg_.mean.push_back(v.get<float>());
  }
  for (auto& v : args["std"]) {
    arg_.std.push_back(v.get<float>());
  }
  arg_.to_rgb = args.value("to_rgb", true);
  arg_.to_float = args.value("to_float", true);
  // assert `mean` is 0 and `std` is 1 when `to_float` is false
  if (!arg_.to_float) {
    for (int i = 0; i < arg_.mean.size(); ++i) {
      if ((int)arg_.mean[i] != 0 || (int)arg_.std[i] != 1) {
        MMDEPLOY_ERROR("mean {} and std {} are not supported in int8 case", arg_.mean, arg_.std);
        throw_exception(eInvalidArgument);
      }
    }
  }
}

/**
  input:
  {
    "ori_img": Mat,
    "img": Tensor,
    "attribute": "",
    "img_shape": [int],
    "ori_shape": [int],
    "img_fields": [int]
  }
  output:
  {
    "img": Tensor,
    "attribute": "",
    "img_shape": [int],
    "ori_shape": [int],
    "img_fields": [string],
    "img_norm_cfg": {
      "mean": [float],
      "std": [float],
      "to_rgb": true
    }
  }
 */

Result<Value> NormalizeImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));

  // copy input data, and update its properties later
  Value output = input;

  auto img_fields = GetImageFields(input);
  for (auto& key : img_fields) {
    Tensor tensor = input[key].get<Tensor>();
    auto desc = tensor.desc();
    assert(desc.data_type == DataType::kINT8 || desc.data_type == DataType::kFLOAT);
    assert(desc.shape.size() == 4 /*n, h, w, c*/);
    assert(desc.shape[3] == arg_.mean.size());

    if (arg_.to_float) {
      OUTCOME_TRY(auto dst, NormalizeImage(tensor));
      SetTransformData(output, key, std::move(dst));
    } else {
      if (arg_.to_rgb) {
        OUTCOME_TRY(auto dst, ConvertToRGB(tensor));
        SetTransformData(output, key, std::move(dst));
      }
    }

    for (auto& v : arg_.mean) {
      output["img_norm_cfg"]["mean"].push_back(v);
    }
    for (auto v : arg_.std) {
      output["img_norm_cfg"]["std"].push_back(v);
    }
    output["img_norm_cfg"]["to_rgb"] = arg_.to_rgb;

    // trace static info & runtime args
    if (output.contains("__tracer__")) {
      output["__tracer__"].get_ref<Tracer&>().Normalize(arg_.mean, arg_.std, arg_.to_rgb,
                                                        desc.data_type);
    }
  }
  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

Normalize::Normalize(const Value& args, int version) : Transform(args) {
  auto impl_creator = gRegistry<NormalizeImpl>().Get(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'Normalize' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Normalize' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (Normalize, 0), [](const Value& config) {
  return std::make_unique<Normalize>(config, 0);
});

MMDEPLOY_DEFINE_REGISTRY(NormalizeImpl);

}  // namespace mmdeploy
