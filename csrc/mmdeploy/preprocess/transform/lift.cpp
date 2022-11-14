// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/lift.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {
Lift::Lift(const Value& args, int version) : Transform(args) {
  std::string type = "Compose";
  auto creator = gRegistry<Transform>().Get(type, version);
  if (!creator) {
    MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                   gRegistry<Transform>().List());
    throw_exception(eEntryNotFound);
  }
  compose_ = creator->Create(args);
}

Result<Value> Lift::Process(const Value& input) {
  Value output;
  for (int i = 0; i < input.size(); i++) {
    Value single = input[i];
    OUTCOME_TRY(auto t, compose_->Process(single));
    output.push_back(std::move(t));
  }
  return std::move(output);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (Lift, 0), [](const Value& config) {
  return std::make_unique<Lift>(config, 0);
});

}  // namespace mmdeploy
