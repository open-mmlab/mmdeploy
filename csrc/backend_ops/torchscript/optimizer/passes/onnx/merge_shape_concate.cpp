// Copyright (c) OpenMMLab. All rights reserved.
#include "merge_shape_concate.h"

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

void MergeShapeConcate(Node* node) {
  auto inputs = node->inputs();

  std::vector<long> gather_value;
  Value* shape_from = nullptr;

  std::vector<Node*> node_to_remove{node};

  // check pattern shape->gather->unsqueeze->concate
  for (auto input : inputs) {
    auto unsqueeze_node = input->node();
    if (unsqueeze_node->kind() != onnx::Unsqueeze || unsqueeze_node->output()->uses().size() != 1)
      return;

    auto axes = unsqueeze_node->is(attr::axes);
    if (axes.size() != 1 && axes[0] != 0) return;

    auto gather_node = unsqueeze_node->input()->node();
    if (gather_node->kind() != onnx::Gather || gather_node->i(attr::axis) != 0 ||
        gather_node->output()->uses().size() != 1)
      return;

    auto gather_inputs = gather_node->inputs();
    auto gather_data = gather_inputs[0];
    auto gather_indices = gather_inputs[1];
    auto shape_node = gather_data->node();
    if (shape_node->kind() != onnx::Shape || shape_node->output()->uses().size() != 1) return;

    auto current_shape_from = shape_node->input();
    if (!shape_from) {
      shape_from = current_shape_from;
    } else {
      if (shape_from != current_shape_from) return;
    }

    auto constant_node = gather_indices->node();
    if (constant_node->kind() != onnx::Constant) return;

    auto gather_indices_val = constant_node->t(attr::value);
    long* data_ptr = gather_indices_val.data_ptr<long>();
    if (gather_indices_val.dim() == 0) {
      gather_value.push_back(data_ptr[0]);
    } else {
      int element_size = gather_indices_val.element_size();
      for (int j = 0; j < element_size; ++j) {
        gather_value.push_back(data_ptr[j]);
      }
    }

    node_to_remove.insert(node_to_remove.end(), {unsqueeze_node, gather_node, shape_node});
  }

  // create constant value
  auto graph = node->owningGraph();
  auto const_node = graph->create(onnx::Constant);
  const_node->t_(attr::value, at::tensor(gather_value));
  auto first_node = node->owningGraph()->block()->nodes().front();
  if (const_node != first_node) const_node->insertBefore(first_node);

  // recreate shape node
  auto shape_node = graph->create(onnx::Shape, {shape_from});
  shape_node->insertBefore(node);

  // create gather node
  auto gather_node = graph->create(onnx::Gather, {shape_node->output(), const_node->output()});

  // insert into graph
  gather_node->insertAfter(node);
  node->output()->replaceAllUsesWith(gather_node->output());

  for (auto n : node_to_remove) {
    n->destroy();
  }
}

void MergeShapeConcate(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      MergeShapeConcate(block);
    }

    if (node->kind() == onnx::Concat) {
      MergeShapeConcate(node);
    }
  }
}

void MergeShapeConcate(const std::shared_ptr<Graph>& graph) { MergeShapeConcate(graph->block()); }

}  // namespace torch_jit
}  // namespace mmdeploy
