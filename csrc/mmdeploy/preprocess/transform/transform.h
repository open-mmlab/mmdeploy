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
  virtual Result<void> Apply(Value& input) = 0;
};

MMDEPLOY_DECLARE_REGISTRY(Transform, std::unique_ptr<Transform>(const Value& config));

std::vector<std::string> GetImageFields(const Value& input);

}  // namespace transform

using transform::Transform;

}  // namespace mmdeploy

#endif  // MMDEPLOY_TRANSFORM_H
