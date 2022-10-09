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

class PipelineCreator : public Creator<Builder> {
 public:
  const char* GetName() const override { return "Pipeline"; }
  unique_ptr<Builder> Create(const Value& config) override {
    return std::make_unique<PipelineBuilder>(config);
  }
};
REGISTER_MODULE(Builder, PipelineCreator);

}  // namespace mmdeploy::graph
