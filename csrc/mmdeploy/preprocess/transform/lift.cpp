// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/lift.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

Lift::Lift(const Value& args) {
  const char* type = "compose";
  if (auto creator = gRegistry<Transform>().Get(type)) {
    compose_ = creator->Create(args);
  } else {
    MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                   gRegistry<Transform>().List());
    throw_exception(eEntryNotFound);
  }
}

Result<void> Lift::Apply(Value& data) {
  for (auto& item : data.array()) {
    OUTCOME_TRY(compose_->Apply(item));
  }
  return success();
}

MMDEPLOY_REGISTER_TRANSFORM(Lift);

}  // namespace mmdeploy
