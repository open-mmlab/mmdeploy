// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PREPROCESS_TRANSFORM_COMPOSE_H_
#define MMDEPLOY_SRC_PREPROCESS_TRANSFORM_COMPOSE_H_

#include "mmdeploy/core/profiler.h"
#include "transform.h"

namespace mmdeploy {

class MMDEPLOY_API Compose : public Transform {
 public:
  explicit Compose(const Value& args, int version = 0);
  ~Compose() override = default;

  Result<Value> Process(const Value& input) override;

 private:
  std::vector<std::unique_ptr<Transform>> transforms_;
  Stream stream_;
  std::vector<profiler::Scope*> transform_scopes_;
  profiler::Scope* scope_{nullptr};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_PREPROCESS_TRANSFORM_COMPOSE_H_
