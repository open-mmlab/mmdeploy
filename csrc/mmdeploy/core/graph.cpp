// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/graph.h"

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"

namespace mmdeploy {
namespace graph {

Result<void> NodeParser::Parse(const Value& config, Node& node) {
  try {
    from_value(config["input"], node.inputs_);
    from_value(config["output"], node.outputs_);
    node.name_ = config.value<std::string>("name", "");
    return success();
  } catch (const Exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", config);
    return failure(e.code());
  }
}

}  // namespace graph

MMDEPLOY_DEFINE_REGISTRY(graph::Node);

MMDEPLOY_DEFINE_REGISTRY(TypeErasedScheduler<Value>);

}  // namespace mmdeploy
