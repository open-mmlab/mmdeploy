// Copyright (c) OpenMMLab. All rights reserved.
#include "onnx_peephole.h"

#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <vector>

#include "utils.h"

namespace mmdeploy {
namespace torch_jit {

using c10::Symbol;
using torch::jit::Block;
using torch::jit::IValue;
using torch::jit::Node;
using torch::jit::TensorType;
using torch::jit::Value;

void RemoveReshapeChain(Node* node) {
  // reshape->reshape => reshape
  auto output = node->output();
  if (!(output->hasUses())) {
    return;
  }
  auto uses = output->uses();

  for (auto use : uses) {
    if (!is_kind(use.user, "onnx::Reshape") || use.offset != 0) {
      return;
    }
  }

  auto input = node->inputs()[0];
  output->replaceAllUsesWith(input);

  node->destroy();
}

void RemoveRedundantCast(Node* node) {
  // Cast(type n)->Cast(type n) => Cast(type n)

  auto to_type = node->i(Symbol::attr("to"));
  auto input = node->input();

  auto input_node = input->node();
  if (is_kind(input_node, "onnx::Cast") && input_node->i(Symbol::attr("to")) == to_type) {
    auto output = node->output();

    output->replaceAllUsesWith(input);
    node->destroy();
  }
}

void ONNXPeephole(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      ONNXPeephole(block);
    }

    if (is_kind(node, "onnx::Reshape")) {
      RemoveReshapeChain(node);
    } else if (is_kind(node, "onnx::Cast")) {
      RemoveRedundantCast(node);
    }
  }
}

void ONNXPeephole(const std::shared_ptr<Graph>& graph) {
  ONNXPeephole(graph->block());
  torch::jit::EliminateDeadCode(
      graph->block(), true,
      torch::jit::DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

}  // namespace torch_jit
}  // namespace mmdeploy
