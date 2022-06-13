// Copyright (c) OpenMMLab. All rights reserved.

#include "normalize.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"

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

    OUTCOME_TRY(auto dst, NormalizeImage(tensor));
    SetTransformData(output, key, std::move(dst));

    for (auto& v : arg_.mean) {
      output["img_norm_cfg"]["mean"].push_back(v);
    }
    for (auto v : arg_.std) {
      output["img_norm_cfg"]["std"].push_back(v);
    }
    output["img_norm_cfg"]["to_rgb"] = arg_.to_rgb;
  }
  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

Normalize::Normalize(const Value& args, int version) : Transform(args) {
  auto impl_creator = Registry<NormalizeImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'Normalize' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Normalize' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class NormalizeCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "Normalize"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override { return make_unique<Normalize>(args, version_); }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, NormalizeCreator);

MMDEPLOY_DEFINE_REGISTRY(NormalizeImpl);

}  // namespace mmdeploy
