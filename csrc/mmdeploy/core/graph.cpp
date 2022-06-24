// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/graph.h"

#include "mmdeploy/core/registry.h"
#include "mmdeploy/graph/common.h"

namespace mmdeploy {
namespace graph {

Result<void> Builder::SetInputs() {
  MMDEPLOY_INFO("{}", config_);
  OUTCOME_TRY(inputs_, ParseStringArray(config_["input"]));
  return success();
}

Result<void> Builder::SetOutputs() {
  MMDEPLOY_INFO("{}", config_);
  OUTCOME_TRY(outputs_, ParseStringArray(config_["output"]));
  return success();
}

Builder::Builder(Value config) : config_(std::move(config)) {
  name_ = config_.value<std::string>("name", "");
}

Result<unique_ptr<Node>> Builder::Build() {
  OUTCOME_TRY(SetInputs());
  OUTCOME_TRY(SetOutputs());
  return BuildImpl();
}

Result<unique_ptr<Builder>> Builder::CreateFromConfig(const Value& config) {
  MMDEPLOY_WARN("config: {}", config);
  auto type = config.value<string>("type", "");
  auto cfg = config;
  // backward compatibility
  if (type.empty()) {
    if (config.contains("pipeline")) {
      type = "Pipeline";
      cfg = config["pipeline"];
      if (config.contains("context")) {
        cfg["context"] = config["context"];
      }
    }
  }
  auto creator = Registry<Builder>::Get().GetCreator(type);
  if (!creator) {
    MMDEPLOY_ERROR("failed to find node creator: {}", type);
    return Status(eEntryNotFound);
  }
  auto builder = creator->Create(cfg);
  if (!builder) {
    MMDEPLOY_ERROR("failed to create node builder: {}", type);
    return Status(eFail);
  }
  return std::move(builder);
}


}  // namespace graph

MMDEPLOY_DEFINE_REGISTRY(graph::Builder);

MMDEPLOY_DEFINE_REGISTRY(TypeErasedScheduler<Value>);

}  // namespace mmdeploy
