// Copyright (c) OpenMMLab. All rights reserved.

#include "inference.h"

#include "archive/json_archive.h"
#include "core/model.h"

namespace mmdeploy::async {

Result<unique_ptr<Pipeline>> InferenceParser::Parse(const Value& config) {
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

    // transfer basic configs
    pipeline_config["input"] = config["input"];
    pipeline_config["output"] = config["output"];
    pipeline_config["name"] = config["name"];

    PipelineParser parser;
    return parser.Parse(pipeline_config);

  } catch (const Exception& e) {
    MMDEPLOY_ERROR("exception: {}", e.what());
    return failure(e.code());
  }
}

}  // namespace mmdeploy::async