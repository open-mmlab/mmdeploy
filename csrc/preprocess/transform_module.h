// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_MODULE_TRANSFORM_MODULE_H_
#define MMDEPLOY_SRC_MODULE_TRANSFORM_MODULE_H_

#include "core/value.h"

namespace mmdeploy {

class Transform;

class TransformModule {
 public:
  ~TransformModule();
  explicit TransformModule(const Value& args);
  TransformModule(TransformModule&&) = default;
  Result<Value> operator()(const Value& input);

 private:
  std::unique_ptr<Transform> transform_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_MODULE_TRANSFORM_MODULE_H_
