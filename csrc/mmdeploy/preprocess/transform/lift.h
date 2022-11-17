// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PREPROCESS_TRANSFORM_LIFT_H_
#define MMDEPLOY_SRC_PREPROCESS_TRANSFORM_LIFT_H_

#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy {

class Lift : public Transform {
 public:
  explicit Lift(const Value& args);
  ~Lift() override = default;

  Result<void> Apply(Value& data) override;

 private:
  std::unique_ptr<Transform> compose_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_PREPROCESS_TRANSFORM_Lift_H_
