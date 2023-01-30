// Copyright (c) OpenMMLab. All rights reserved.
#include "merge_shape_concate.h"

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

void MergeShapeConcate(Node* node) {
  auto inputs = node->inputs();

  std::vector<int64_t> gather_value;
  Value* shape_from = nullptr;

  std::vector<Node*> node_to_remove{node};

  // check pattern shape->gather->unsqueeze->concate
  for (auto input : inputs) {
    auto unsqueeze_node = input->node();
    if (!is_kind(unsqueeze_node, "onnx::Unsqueeze") || unsqueeze_node->output()->uses().size() != 1)
      return;

    auto axes = unsqueeze_node->is(Symbol::attr("axes"));
    if (axes.size() != 1 && axes[0] != 0) return;

    auto gather_node = unsqueeze_node->input()->node();
    if (!is_kind(gather_node, "onnx::Gather") || gather_node->i(Symbol::attr("axis")) != 0 ||
        gather_node->output()->uses().size() != 1)
      return;

    auto gather_inputs = gather_node->inputs();
    auto gather_data = gather_inputs[0];
    auto gather_indices = gather_inputs[1];
    auto shape_node = gather_data->node();
    if (!is_kind(shape_node, "onnx::Shape") || shape_node->output()->uses().size() != 1) return;

    auto current_shape_from = shape_node->input();
    if (!shape_from) {
      shape_from = current_shape_from;
    } else {
      if (shape_from != current_shape_from) return;
    }

    auto constant_node = gather_indices->node();
    if (!is_kind(constant_node, "onnx::Constant")) return;

    auto gather_indices_val = constant_node->t(Symbol::attr("value"));
    int64_t* data_ptr = gather_indices_val.data_ptr<int64_t>();
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
  auto const_node = graph->create(Symbol::onnx("Constant"));
  const_node->t_(Symbol::attr("value"), at::tensor(gather_value));
  auto first_node = node->owningGraph()->block()->nodes().front();
  if (const_node != first_node) const_node->insertBefore(first_node);

  // recreate shape node
  auto shape_node = graph->create(Symbol::onnx("Shape"), {shape_from});
  shape_node->insertBefore(node);

  // create gather node
  auto gather_node =
      graph->create(Symbol::onnx("Gather"), {shape_node->output(), const_node->output()});

  // insert into graph
  gather_node->insertAfter(node);
  node->output()->replaceAllUsesWith(gather_node->output());

  for (auto n : node_to_remove) {
    n->destroy();
  }
}

void MergeShapeConcate(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      MergeShapeConcate(block);
    }

    if (is_kind(node, "onnx::Concat")) {
      MergeShapeConcate(node);
    }
  }
}

void MergeShapeConcate(const std::shared_ptr<Graph>& graph) { MergeShapeConcate(graph->block()); }

}  // namespace torch_jit
}  // namespace mmdeploy
