// https://github.com/pytorch/pytorch/blob/v1.8.1/torch/csrc/jit/passes/common_subexpression_elimination.cpp
#include "common_subgraph_elimination.h"

#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

namespace mmdeploy {
namespace torch_jit {

using c10::Symbol;
using torch::jit::Block;
using torch::jit::EqualNode;
using torch::jit::HashNode;
using torch::jit::Node;
using torch::jit::Value;

struct EqualNodeWithParams {
  EqualNodeWithParams(std::unordered_map<std::string, Tensor>& params) : params_(params) {}

  bool operator()(const Node* lhs, const Node* rhs) const {
    auto lhs_inputs = lhs->inputs();
    auto rhs_inputs = rhs->inputs();
  }

 private:
  std::unordered_map<std::string, Tensor>& params_;
};

struct CommonSubexpressionEliminator {
  using ParamMapType = std::unordered_map<std::string, std::pair<Tensor, Value*>>;
  CommonSubexpressionEliminator(std::shared_ptr<Graph> graph,
                                std::unordered_map<std::string, Tensor>& params)
      : graph_(std::move(graph)), params_(params) {}

  bool run(std::function<Node*(Node*)> parent_lookup_fn) {
    ParamMapType param_map;
    return run(graph_->block(), std::move(parent_lookup_fn), param_map);
  }

  // The function implements common subexpression elimination.
  // Since the nodes are visited in topological order, one pass is enough.
  // returns true if CSE made changes to a graph
  bool run(Block* block, std::function<Node*(Node*)> parent_lookup_fn, ParamMapType& param_map) {
    std::unordered_set<Node*, HashNode, EqualNode> subexprs;
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto node = *it;

      // check if inputs come from params(graph input)
      auto node_inputs = node->inputs();
      for (auto input : node_inputs) {
        if (input->node()->kind() == Symbol::fromQualString("prim::Param")) {
          auto debug_name = input->debugName();

          // check if input in params_
          if (params_.find(debug_name) == params_.end()) continue;

          // check if input is already visited.
          if (param_map.find(debug_name) != param_map.end()) continue;

          // check if there is a param has same value with input
          auto val = params_[debug_name];
          bool update_map = true;
          for (auto kv : param_map) {
            auto param_val = kv.second.first;
            if (val.device() != param_val.device()) continue;
            if (val.dtype() != param_val.dtype()) continue;
            if (!val.equal(param_val)) continue;
            input->replaceAllUsesWith(kv.second.second);
            update_map = false;
            break;
          }

          // add input to param_map
          if (update_map) {
            param_map.emplace(debug_name,
                              std::make_pair<Tensor, Value*>(std::move(val), std::move(input)));
          }
        }
      }

      if (!node->blocks().empty()) {
        // Traverse sub-blocks.
        for (auto block : node->blocks()) {
          changed |= run(
              block,
              [&](Node* n) {
                auto existing = subexprs.find(n);
                if (existing != subexprs.end()) {
                  return *existing;
                }

                return parent_lookup_fn(n);
              },
              param_map);
        }

        continue;
      }

      // Check for CSE opportunities in the parent block.
      auto parent_lookup = parent_lookup_fn(node);
      auto g_out = node->owningGraph()->outputs();
      if (parent_lookup != nullptr) {
        changed = true;
        node->replaceAllUsesWith(parent_lookup);
        it.destroyCurrent();
        continue;
      }

      // Check whether the same subexpression already exists.
      auto subit = subexprs.insert(node);
      if (!subit.second) {
        // Subexpression exists, replace the uses of node, and destroy it.
        auto existing = *subit.first;

        changed = true;
        node->replaceAllUsesWith(existing);
        // Destroy the node.
        it.destroyCurrent();
      }
    }

    return changed;
  }

 private:
  std::shared_ptr<Graph> graph_;
  std::unordered_map<std::string, Tensor>& params_;
};

void CommonSubgraphElimination(std::shared_ptr<Graph>& graph,
                               std::unordered_map<std::string, Tensor>& params) {
  CommonSubexpressionEliminator cse(graph, params);
  cse.run([](Node*) { return nullptr; });
}
}  // namespace torch_jit
}  // namespace mmdeploy
