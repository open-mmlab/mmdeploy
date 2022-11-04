// Copyright (c) OpenMMLab. All rights reserved.

#include "transform.h"

#include "mmdeploy/core/registry.h"

namespace mmdeploy::transform {

Result<Value> Transform::Process(const Value& input) {
  auto output = input;
  OUTCOME_TRY(Apply(output));
  return output;
}

MMDEPLOY_DEFINE_REGISTRY(Transform);

}  // namespace mmdeploy::transform
