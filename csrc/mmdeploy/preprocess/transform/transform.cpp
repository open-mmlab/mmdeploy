// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/transform.h"

#include "mmdeploy/core/registry.h"

namespace mmdeploy::transform {

Result<Value> Transform::Process(const Value& input) {
  auto output = input;
  {
    operation::Session session;
    OUTCOME_TRY(Apply(output));
    for (const auto& buffer : session.buffers()) {
      output["__data__"].push_back(buffer);
    }
  }
  return output;
}

operation::Context GetContext(const Value& config) {
  if (config.contains("context")) {
    auto device = config["context"]["device"].get<Device>();
    auto stream = config["context"]["stream"].get<Stream>();
    return {device, stream};
  }
  throw_exception(eInvalidArgument);
};

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
