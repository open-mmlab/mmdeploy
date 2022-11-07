// Copyright (c) OpenMMLab. All rights reserved.

#include "transform.h"

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

MMDEPLOY_DEFINE_REGISTRY(Transform);

}  // namespace mmdeploy::transform
