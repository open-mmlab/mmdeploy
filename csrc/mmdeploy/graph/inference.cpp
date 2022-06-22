// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/graph/common.h"
#include "mmdeploy/graph/pipeline.h"

namespace mmdeploy::graph {

class InferenceCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Inference"; }
  std::unique_ptr<Node> Create(const Value& config) override {
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

    return BuildFromConfig<PipelineBuilder>(pipeline_config).value();
  }
};

REGISTER_MODULE(Node, InferenceCreator);

}  // namespace mmdeploy::graph
