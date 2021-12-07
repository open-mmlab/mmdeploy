// Copyright (c) OpenMMLab. All rights reserved.

#include "graph/inference.h"

#include "archive/json_archive.h"
#include "archive/value_archive.h"
#include "core/operator.h"
#include "graph/common.h"

namespace mmdeploy::graph {

unique_ptr<Inference> Inference::Create(const Value& param) {
  try {
    auto inst = std::make_unique<Inference>();
    auto& model_value = param["params"]["model"];
    if (model_value.is_any<Model>()) {
      inst->model_ = model_value.get<Model>();
    } else if (model_value.is_string()) {
      auto model_path = model_value.get<std::string>();
      inst->model_ = Model(model_path);
    } else {
      ERROR("unsupported model specification");
      return nullptr;
    }

    auto pipeline_json = inst->model_.ReadFile("pipeline.json").value();
    auto json = nlohmann::json::parse(pipeline_json);

    auto context = param.value("context", Value(ValueType::kObject));
    context["model"] = inst->model_;

    auto value = from_json<Value>(json);
    value["context"] = context;
    inst->pipeline_ = Pipeline::Create(value);

    if (!inst->pipeline_) {
      return nullptr;
    }

    from_value(param["input"], inst->inputs_);
    from_value(param["output"], inst->outputs_);

    return inst;
  } catch (const std::exception& e) {
    ERROR("unhandled exception: {}", e.what());
  }
  return nullptr;
}

void Inference::Build(TaskGraph& graph) {
  auto enter = graph.Add([this](Context& ctx) -> Result<void> {
    OUTCOME_TRY(auto args, Keys2Idxs(ctx.current(), inputs_));
    ctx.push(std::move(args));
    return success();
  });
  enter->set_name("inference/enter");

  pipeline_->Build(graph);

  auto exit = graph.Add([this](Context& ctx) -> Result<void> {
    auto rets = ctx.pop();
    OUTCOME_TRY(Idxs2Keys(std::move(rets), outputs_, ctx.current()));
    return success();
  });
  exit->set_name("inference/exit");
}

class InferenceNodeCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Inference"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override { return Inference::Create(value); }
};

REGISTER_MODULE(Node, InferenceNodeCreator);

}  // namespace mmdeploy::graph
