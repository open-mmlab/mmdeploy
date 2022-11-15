// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/lift.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {
Lift::Lift(const Value& args, int version) : Transform(args) {
  std::string type = "Compose";
  auto creator = Registry<Transform>::Get().GetCreator(type, version);
  if (!creator) {
    MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                   Registry<Transform>::Get().List());
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

class LiftCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "Lift"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override { return std::make_unique<Lift>(args, version_); }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, LiftCreator);
}  // namespace mmdeploy
