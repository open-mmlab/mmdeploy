// Copyright (c) OpenMMLab. All rights reserved.

#include "graph/pipeline.h"

#include "archive/value_archive.h"
#include "core/operator.h"
#include "graph/common.h"

namespace mmdeploy::graph {

unique_ptr<Pipeline> Pipeline::Create(const Value& config) {
  try {
    auto inst = std::make_unique<Pipeline>();
    from_value(config["pipeline"]["input"], inst->inputs_);
    from_value(config["pipeline"]["output"], inst->outputs_);
    for (auto task_config : config["pipeline"]["tasks"]) {
      auto name = task_config.value("name", std::string{});
      auto type = task_config.value("type", std::string{});
      if (config.contains("context")) {
        //        ERROR("passing context: {}", config["context"]);
        task_config["context"].update(config["context"]);
      }
      if (auto node = CreateFromRegistry<Node>(task_config); node) {
        inst->nodes_.push_back(std::move(node).value());
        //      } else if (auto task = Task::Create(task_config); task) {
        //        inst->nodes_.push_back(std::move(task));
      } else {
        ERROR("could not create {}:{}", name, type);
        return nullptr;
      }
    }
    return inst;
  } catch (...) {
    return nullptr;
  }
}

void Pipeline::Build(TaskGraph& graph) {
  auto enter = graph.Add([this](Context& ctx) -> Result<void> {
    auto args = ctx.pop();
    ctx.push(ValueType::kObject);
    OUTCOME_TRY(Idxs2Keys(std::move(args), inputs_, ctx.current()));
    return success();
  });
  enter->set_name("pipeline/enter");
  for (const auto& node : nodes_) {
    node->Build(graph);
  }
  auto exit = graph.Add([this](Context& ctx) -> Result<void> {
    auto rets = ctx.pop();
    ctx.push(ValueType::kArray);
    OUTCOME_TRY(Keys2Idxs(std::move(rets), outputs_, ctx.current()));
    return success();
  });
  exit->set_name("pipeline/exit");
}

class PipelineCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Pipeline"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override { return Pipeline::Create(value); }
};

REGISTER_MODULE(Node, PipelineCreator);

}  // namespace mmdeploy::graph
