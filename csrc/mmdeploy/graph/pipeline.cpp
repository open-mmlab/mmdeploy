// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/pipeline.h"

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/graph/static_router.h"

namespace mmdeploy::graph {

PipelineBuilder::PipelineBuilder(Value config) : Builder(std::move(config)) {}

Result<unique_ptr<Node>> PipelineBuilder::BuildImpl() {
  // create static router
  return StaticRouterBuilder{}.Build(config_).value();
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Builder, (Pipeline, 0), [](const Value& config) {
  return std::make_unique<PipelineBuilder>(config);
});

}  // namespace mmdeploy::graph
