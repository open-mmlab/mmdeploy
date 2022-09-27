// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/inference.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/graph/common.h"

namespace mmdeploy::graph {

using namespace framework;

InferenceBuilder::InferenceBuilder(Value config) : Builder(std::move(config)) {}

Result<unique_ptr<Node>> InferenceBuilder::BuildImpl() {
  auto& model_config = config_["params"]["model"];
  Model model;
  if (model_config.is_any<Model>()) {
    model = model_config.get<Model>();
  } else {
    auto model_name = model_config.get<string>();
    if (auto m = Maybe{config_} / "context" / "model" / model_name / identity<Model>{}) {
      model = *m;
    } else {
      model = Model(model_name);
    }
  }
  auto pipeline_json = model.ReadFile("pipeline.json").value();
  auto json = nlohmann::json::parse(pipeline_json);

  auto context = config_.value("context", Value(ValueType::kObject));
  context["model"] = std::move(model);

  auto pipeline_config = from_json<Value>(json);
  pipeline_config["context"] = context;

  MMDEPLOY_INFO("{}", pipeline_config);

  OUTCOME_TRY(auto pipeline_builder, Builder::CreateFromConfig(pipeline_config));
  OUTCOME_TRY(auto node, pipeline_builder->Build());

  OUTCOME_TRY(CheckInputs(*pipeline_builder));
  OUTCOME_TRY(CheckOutputs(*pipeline_builder));

  return std::move(node);
}
Result<void> InferenceBuilder::CheckInputs(Builder& builder) {
  OUTCOME_TRY(auto inputs_internal, ParseStringArray(config_["input"]));
  MMDEPLOY_INFO("{} <- {}", builder.inputs(), inputs_internal);
  if (builder.inputs().size() != inputs_internal.size()) {
    MMDEPLOY_ERROR("mis-matched number of inputs: {} vs {}", builder.inputs().size(),
                   inputs_internal.size());
    return Status(eInvalidArgument);
  }
  return success();
}

Result<void> InferenceBuilder::CheckOutputs(Builder& builder) {
  OUTCOME_TRY(auto outputs_internal, ParseStringArray(config_["output"]));
  MMDEPLOY_INFO("{} -> {}", builder.outputs(), outputs_internal);
  if (builder.outputs().size() != outputs_internal.size()) {
    MMDEPLOY_ERROR("mis-matched number of outputs: {} vs {}", builder.outputs().size(),
                   outputs_internal.size());
    return Status(eInvalidArgument);
  }
  return success();
}

class InferenceCreator : public Creator<Builder> {
 public:
  const char* GetName() const override { return "Inference"; }
  unique_ptr<Builder> Create(const Value& config) override {
    return std::make_unique<InferenceBuilder>(config);
  }
};

REGISTER_MODULE(Builder, InferenceCreator);

}  // namespace mmdeploy::graph
