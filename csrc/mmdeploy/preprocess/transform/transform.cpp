// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/transform.h"

#include "mmdeploy/core/registry.h"

namespace mmdeploy::transform {

std::vector<std::string> GetImageFields(const Value& input) {
  if (input.contains("img_fields")) {
    if (input["img_fields"].is_string()) {
      return {input["img_fields"].get<std::string>()};
    } else if (input["img_fields"].is_array()) {
      std::vector<std::string> img_fields;
      for (auto& v : input["img_fields"]) {
        img_fields.push_back(v.get<std::string>());
      }
      return img_fields;
    }
  }
  return {"img"};
};

MMDEPLOY_DEFINE_REGISTRY(Transform);

}  // namespace mmdeploy::transform
