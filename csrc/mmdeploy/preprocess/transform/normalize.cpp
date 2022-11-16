// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

class Normalize : public Transform {
 public:
  explicit Normalize(const Value& args) {
    if (!args.contains("mean") || !args.contains("std")) {
      MMDEPLOY_ERROR("no 'mean' or 'std' is configured");
      throw std::invalid_argument("no 'mean' or 'std' is configured");
    }
    for (auto& v : args["mean"]) {
      mean_.push_back(v.get<float>());
    }
    for (auto& v : args["std"]) {
      std_.push_back(v.get<float>());
    }
    to_rgb_ = args.value("to_rgb", true);

    // auto context = GetContext(args);
    normalize_ = operation::Managed<operation::Normalize>::Create(
        operation::Normalize::Param{mean_, std_, to_rgb_});
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

  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));

    auto img_fields = GetImageFields(input);
    for (auto& key : img_fields) {
      Tensor tensor = input[key].get<Tensor>();
      auto desc = tensor.desc();
      assert(desc.data_type == DataType::kINT8 || desc.data_type == DataType::kFLOAT);
      assert(desc.shape.size() == 4 /*n, h, w, c*/);
      assert(desc.shape[3] == mean_.size());

      Tensor dst;
      OUTCOME_TRY(normalize_.Apply(tensor, dst));
      input[key] = std::move(dst);

      for (auto& v : mean_) {
        input["img_norm_cfg"]["mean"].push_back(v);
      }
      for (auto v : std_) {
        input["img_norm_cfg"]["std"].push_back(v);
      }
      input["img_norm_cfg"]["to_rgb"] = to_rgb_;

      // trace static info & runtime args
      if (input.contains("__tracer__")) {
        input["__tracer__"].get_ref<Tracer&>().Normalize(mean_, std_, to_rgb_, desc.data_type);
      }
    }
    MMDEPLOY_DEBUG("output: {}", to_json(input).dump(2));
    return success();
  }

 private:
  operation::Managed<operation::Normalize> normalize_;
  std::vector<float> mean_;
  std::vector<float> std_;
  bool to_rgb_;
};

MMDEPLOY_REGISTER_TRANSFORM(Normalize);

}  // namespace mmdeploy::transform
