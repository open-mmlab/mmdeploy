// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_MODULE_TRANSFORM_MODULE_H_
#define MMDEPLOY_SRC_MODULE_TRANSFORM_MODULE_H_

#include "mmdeploy/core/value.h"

namespace mmdeploy {

class Transform;

class TransformModule {
 public:
  ~TransformModule();
  TransformModule(TransformModule&&) noexcept;

  explicit TransformModule(const Value& args);
  Result<Value> operator()(const Value& input);

 private:
  int pipeline_id_;
  int node_id_;
  std::unique_ptr<Transform> transform_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_MODULE_TRANSFORM_MODULE_H_
