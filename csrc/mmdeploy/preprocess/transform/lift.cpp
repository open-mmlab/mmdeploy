// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

class Lift : public Transform {
 public:
  explicit Lift(const Value& args) {
    const char* type = "Compose";
    if (auto creator = gRegistry<Transform>().Get(type)) {
      compose_ = creator->Create(args);
    } else {
      MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                     gRegistry<Transform>().List());
      throw_exception(eEntryNotFound);
    }
  }

  Result<void> Apply(Value& data) override {
    for (auto& item : data.array()) {
      OUTCOME_TRY(compose_->Apply(item));
    }
    return success();
  }

 private:
  std::unique_ptr<Transform> compose_;
};

MMDEPLOY_REGISTER_TRANSFORM(Lift);

}  // namespace mmdeploy::transform
