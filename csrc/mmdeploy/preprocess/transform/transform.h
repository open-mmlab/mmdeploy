// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_H
#define MMDEPLOY_TRANSFORM_H

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/operation/operation.h"

namespace mmdeploy {

using namespace framework;

namespace transform {

class MMDEPLOY_API Transform {
 public:
  virtual ~Transform() = default;
  virtual Result<void> Apply(Value& data) = 0;
};

MMDEPLOY_API std::vector<std::string> GetImageFields(const Value& input);

MMDEPLOY_DECLARE_REGISTRY(Transform, std::unique_ptr<Transform>(const Value& config));

#define MMDEPLOY_REGISTER_TRANSFORM2(type, desc)                                                   \
  MMDEPLOY_REGISTER_FACTORY_FUNC(::mmdeploy::transform::Transform, desc, [](const Value& config) { \
    return std::make_unique<type>(config);                                                         \
  });

#define MMDEPLOY_REGISTER_TRANSFORM(type) MMDEPLOY_REGISTER_TRANSFORM2(type, (type, 0))

}  // namespace transform

using transform::Transform;

}  // namespace mmdeploy

#endif  // MMDEPLOY_TRANSFORM_H
