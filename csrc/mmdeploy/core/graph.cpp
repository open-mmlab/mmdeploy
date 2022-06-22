// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/graph.h"

#include "mmdeploy/core/registry.h"
#include "mmdeploy/graph/common.h"

namespace mmdeploy {
namespace graph {

Result<void> Builder::SetInputs() {
  OUTCOME_TRY(inputs_, ParseStringArray(config_["input"]));
  return success();
}

Result<void> Builder::SetOutputs() {
  OUTCOME_TRY(outputs_, ParseStringArray(config_["output"]));
  return success();
}

Builder::Builder(Value config) : config_(std::move(config)) {
  name_ = config_.value<std::string>("name", "");
}

}  // namespace graph

MMDEPLOY_DEFINE_REGISTRY(graph::Builder);

MMDEPLOY_DEFINE_REGISTRY(TypeErasedScheduler<Value>);

}  // namespace mmdeploy
