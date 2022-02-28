// Copyright (c) OpenMMLab. All rights reserved.

#include "graph/inference.h"

#include "archive/json_archive.h"
#include "archive/value_archive.h"
#include "core/operator.h"
#include "graph/common.h"

namespace mmdeploy::graph {

Inference::Inference(const Value& cfg) : BaseNode(cfg) {
  auto& model_value = cfg["params"]["model"];
  if (model_value.is_any<Model>()) {
    model_ = model_value.get<Model>();
  } else if (model_value.is_string()) {
    auto model_path = model_value.get<std::string>();
    model_ = Model(model_path);
  } else {
    MMDEPLOY_ERROR("unsupported model specification");
    throw_exception(eInvalidArgument);
  }

  auto pipeline_json = model_.ReadFile("pipeline.json").value();
  auto json = nlohmann::json::parse(pipeline_json);

  auto context = cfg.value("context", Value(ValueType::kObject));
  context["model"] = model_;

  auto value = from_json<Value>(json);
  value["context"] = context;
  pipeline_ = std::make_unique<Pipeline>(value);
  if (!pipeline_) {
    MMDEPLOY_ERROR("failed to create pipeline");
    throw_exception(eFail);
  }
}

void Inference::Build(TaskGraph& graph) { pipeline_->Build(graph); }

class InferenceNodeCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Inference"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override {
    return std::make_unique<Inference>(value);
  }
};

REGISTER_MODULE(Node, InferenceNodeCreator);

}  // namespace mmdeploy::graph
