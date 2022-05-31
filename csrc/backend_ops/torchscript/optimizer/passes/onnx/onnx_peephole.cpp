// Copyright (c) OpenMMLab. All rights reserved.
#include "onnx_peephole.h"

#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <vector>

namespace mmdeploy {
namespace torch_jit {

using torch::jit::Block;
using torch::jit::IValue;
using torch::jit::Node;
using torch::jit::TensorType;
using torch::jit::Value;

namespace attr {
using namespace ::c10::attr;
}

namespace onnx {
using namespace ::c10::onnx;
}

void RemoveReshapeChain(Node* node) {
  // reshape->reshape => reshape
  auto output = node->output();
  if (!(output->hasUses())) {
    return;
  }
  auto uses = output->uses();

  for (auto use : uses) {
    if (use.user->kind() != onnx::Reshape || use.offset != 0) {
      return;
    }
  }

  auto input = node->inputs()[0];
  output->replaceAllUsesWith(input);

  node->destroy();
}

void RemoveRedundantCast(Node* node) {
  // Cast(type n)->Cast(type n) => Cast(type n)

  auto to_type = node->i(attr::to);
  auto input = node->input();

  auto input_node = input->node();
  if (input_node->kind() == onnx::Cast && input_node->i(attr::to) == to_type) {
    auto output = node->output();

    output->replaceAllUsesWith(input);
    node->destroy();
  }
}

void ONNXPeephole(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      ONNXPeephole(block);
    }

    if (node->kind() == onnx::Reshape) {
      RemoveReshapeChain(node);
    } else if (node->kind() == onnx::Cast) {
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
