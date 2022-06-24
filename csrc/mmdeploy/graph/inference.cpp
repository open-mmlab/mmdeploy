// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/inference.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/graph/pipeline.h"

namespace mmdeploy::graph {

unique_ptr<Builder> CreateInferenceBuilder(const Value& config) {
  MMDEPLOY_INFO("{}", config);
  auto& model_config = config["params"]["model"];
  Model model;
  if (model_config.is_any<Model>()) {
    model = model_config.get<Model>();
  } else {
    model = Model(model_config.get<string>());
  }
  auto pipeline_json = model.ReadFile("pipeline.json").value();
  auto json = nlohmann::json::parse(pipeline_json);

  auto context = config.value("context", Value(ValueType::kObject));
  context["model"] = std::move(model);

  auto pipeline_config = from_json<Value>(json);
  pipeline_config["context"] = context;

  MMDEPLOY_INFO("{}", pipeline_config);

  return Builder::CreateFromConfig(pipeline_config).value();
  // return std::make_unique<PipelineBuilder>(pipeline_config);
}

class InferenceCreator : public Creator<Builder> {
 public:
  const char* GetName() const override { return "Inference"; }
  unique_ptr<Builder> Create(const Value& config) override {
    return CreateInferenceBuilder(config);
  }
};

REGISTER_MODULE(Builder, InferenceCreator);

// InferenceBuilder::InferenceBuilder(Value config) : Builder(std::move(config)) {}
//
// Result<void> InferenceBuilder::SetInputs() { return success(); }
//
// Result<void> InferenceBuilder::SetOutputs() { return success(); }
//
// Result<unique_ptr<Node>> InferenceBuilder::BuildImpl() { return nullptr; }

}  // namespace mmdeploy::graph
