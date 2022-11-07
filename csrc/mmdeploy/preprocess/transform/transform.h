// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_H
#define MMDEPLOY_TRANSFORM_H

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/preprocess/operation/operation.h"

namespace mmdeploy {

using namespace framework;

namespace transform {

// template <typename Key, typename Val>
// void SetTransformData(Value& dst, Key&& key, Val val) {
//   dst[std::forward<Key>(key)] = val;
//   dst["__data__"].push_back(std::move(val));
// }

inline std::vector<std::string> GetImageFields(const Value& input) {
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
}

inline operation::Context GetContext(const Value& config) {
  if (config.contains("context")) {
    auto device = config["context"]["device"].get<Device>();
    auto stream = config["context"]["stream"].get<Stream>();
    return {device, stream};
  }
  throw_exception(eInvalidArgument);
}

class MMDEPLOY_API Transform {
 public:
  virtual ~Transform() = default;
  virtual Result<void> Apply(Value& input) = 0;

  [[deprecated]] Result<Value> Process(const Value& input);
};

MMDEPLOY_DECLARE_REGISTRY(Transform, std::unique_ptr<Transform>(const Value& config));

}  // namespace transform

using transform::Transform;

}  // namespace mmdeploy

#define MMDEPLOY_REGISTER_TRANSFORM_IMPL(base_type, desc, impl_type) \
  MMDEPLOY_REGISTER_FACTORY_FUNC(                                    \
      base_type, desc, [](const Value& config) { return std::make_unique<impl_type>(config); });

#endif  // MMDEPLOY_TRANSFORM_H
