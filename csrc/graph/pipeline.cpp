// Copyright (c) OpenMMLab. All rights reserved.

#include "graph/pipeline.h"

#include "archive/value_archive.h"
#include "core/operator.h"
#include "graph/common.h"

namespace mmdeploy::graph {

Pipeline::Pipeline(const Value& cfg) : BaseNode(cfg["pipeline"]) {
  input_idx_ = UpdateBindings(inputs(), kWrite);
  for (auto task_config : cfg["pipeline"]["tasks"]) {
    auto name = task_config.value("name", std::string{});
    auto type = task_config.value("type", std::string{});
    if (cfg.contains("context")) {
      task_config["context"].update(cfg["context"]);
    }
    if (auto node = CreateFromRegistry<Node>(task_config); node) {
      nodes_.push_back(std::move(node).value());
      node_input_idx_.push_back(UpdateBindings(nodes_.back()->inputs(), kRead));
      node_output_idx_.push_back(UpdateBindings(nodes_.back()->outputs(), kWrite));
    } else {
      MMDEPLOY_ERROR("could not create {}:{}", name, type);
      throw_exception(eFail);
    }
  }
  output_idx_ = UpdateBindings(outputs(), kRead);
}

void Pipeline::Build(TaskGraph& graph) {
  graph
      .Add([this](Context& ctx) -> Result<void> {
        ctx.current().array().resize(binding_name_to_idx_.size());
        return success();
      })
      ->set_name(fmt::format("{}.call", name()));
  ;
  for (int index = 0; index < nodes_.size(); ++index) {
    graph.Add([this, index](Context& ctx) { return Call(ctx, index); })
        ->set_name(fmt::format("{}.call", nodes_[index]->name()));
    nodes_[index]->Build(graph);
    graph.Add([this, index](Context& ctx) { return Ret(ctx, index); })
        ->set_name(fmt::format("{}.ret", nodes_[index]->name()));
  }
  graph
      .Add([this](Context& ctx) -> Result<void> {
        auto vars = std::move(ctx.current()).array();
        return Gather(std::move(vars), output_idx_, ctx.current().array());
      })
      ->set_name(fmt::format("{}.ret", name()));
}

std::vector<int> Pipeline::UpdateBindings(const vector<std::string>& names, BindingType type) {
  std::vector<int> idxs;
  for (const auto& name : names) {
    auto it = binding_name_to_idx_.lower_bound(name);
    if (it == binding_name_to_idx_.end() || it->first != name) {
      if (type == kRead) {
        MMDEPLOY_ERROR("unknown binding name: {}", name);
        throw_exception(eEntryNotFound);
      } else {
        auto index = static_cast<int>(binding_name_to_idx_.size());
        it = binding_name_to_idx_.emplace_hint(it, name, index);
        binding_idx_to_name_.emplace(index, name);
      }
    }
    idxs.push_back(it->second);
  }
  return idxs;
}

Result<void> Pipeline::Call(Context& ctx, int idx) {
  OUTCOME_TRY(auto&& args, Gather(ctx.current().array(), node_input_idx_[idx]));
  ctx.push(std::move(args));
  return success();
}

Result<void> Pipeline::Ret(Context& ctx, int idx) {
  auto rets = ctx.pop().array();
  return Scatter(std::move(rets), node_output_idx_[idx], ctx.current().array());
}

class PipelineCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Pipeline"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override {
    return std::make_unique<Pipeline>(value);
  }
};

REGISTER_MODULE(Node, PipelineCreator);

}  // namespace mmdeploy::graph
