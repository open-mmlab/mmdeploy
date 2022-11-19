// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PREPROCESS_TRANSFORM_LIFT_H_
#define MMDEPLOY_SRC_PREPROCESS_TRANSFORM_LIFT_H_

#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy {

class MMDEPLOY_API Lift : public Transform {
 public:
  explicit Lift(const Value& args, int version = 0);
  ~Lift() override = default;

  Result<Value> Process(const Value& input) override;

 private:
  std::unique_ptr<Transform> compose_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_PREPROCESS_TRANSFORM_Lift_H_
