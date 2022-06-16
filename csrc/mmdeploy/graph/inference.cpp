// Copyright (c) OpenMMLab. All rights reserved.

#include "inference.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/model.h"

namespace mmdeploy::graph {

Result<unique_ptr<Inference>> InferenceParser::Parse(const Value& config) {
  try {
    auto& model_config = config["params"]["model"];
    Model model;
    if (model_config.is_any<Model>()) {
      model = model_config.get<Model>();
    } else {
      model = Model(model_config.get<string>());
    }
    OUTCOME_TRY(auto pipeline_json, model.ReadFile("pipeline.json"));
    auto json = nlohmann::json::parse(pipeline_json);

    auto context = config.value("context", Value(ValueType::kObject));
    context["model"] = std::move(model);

    auto pipeline_config = from_json<Value>(json);
    pipeline_config["context"] = context;

    auto inference = std::make_unique<Inference>();
    OUTCOME_TRY(NodeParser::Parse(config, *inference));
    OUTCOME_TRY(inference->pipeline_, PipelineParser{}.Parse(pipeline_config));

    return std::move(inference);
  } catch (const Exception& e) {
    MMDEPLOY_ERROR("exception: {}", e.what());
    return failure(e.code());
  }
}

class InferenceCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Inference"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override {
    return InferenceParser::Parse(value).value();
  }
};

REGISTER_MODULE(Node, InferenceCreator);

}  // namespace mmdeploy::graph
