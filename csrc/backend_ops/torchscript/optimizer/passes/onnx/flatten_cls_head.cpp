// Copyright (c) OpenMMLab. All rights reserved.
#include "flatten_cls_head.h"

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <vector>

#include "utils.h"

namespace mmdeploy {
namespace torch_jit {

using c10::Symbol;
using torch::jit::IValue;
using torch::jit::Match;
using torch::jit::TensorType;
using torch::jit::TypeKind;
using torch::jit::Value;

static bool matchClsHead(const Match& match, const std::unordered_map<std::string, Value*>& map) {
  // TODO: check if value map in latest pytorch can ease the filter.

  // check cat -1
  {
    // check if the shape of second inputs is 1
    auto cat_v1 = match.values_map.at(map.at("cat1"));
    if (cat_v1->type()->kind() != TypeKind::TensorType) return false;
    auto cat_v1_type = cat_v1->type()->cast<TensorType>();
    auto cat_v1_size = cat_v1_type->sizes().concrete_sizes();
    if (!cat_v1_size.has_value()) return false;
    IValue cat_v1_size_value(cat_v1_size.value());
    auto size_list = cat_v1_size_value.toIntList();
    if (size_list.size() != 1 || size_list[0] != 1) return false;
  }

  // check unsqueeze
  auto cat_v0 = match.values_map.at(map.at("cat0"));
  auto unsqueeze_node = cat_v0->node();
  {
    if (!is_kind(unsqueeze_node, "onnx::Unsqueeze")) return false;
    auto unsqueeze_axes = unsqueeze_node->is(Symbol::attr("axes"));
    if (unsqueeze_axes.size() != 1 || unsqueeze_axes[0] != 0) return false;
  }

  // check gather
  auto gather_node = unsqueeze_node->input()->node();
  auto gather_inputs = gather_node->inputs();
  {
    if (!is_kind(gather_node, "onnx::Gather")) return false;
    auto gather_axis = gather_node->i(Symbol::attr("axis"));
    if (gather_axis != 0) return false;
  }

  auto x = match.values_map.at(map.at("x"));
  // check shape
  auto shape_node = gather_inputs[0]->node();
  {
    if (!is_kind(shape_node, "onnx::Shape")) return false;
    if (shape_node->input() != x) return false;
  }

  // check constant
  auto const_node = gather_inputs[1]->node();
  {
    if (!is_kind(const_node, "onnx::Constant")) return false;
    auto ival = const_node->t(Symbol::attr("value"));
    if (ival.dim() != 0) return false;
    auto ival_dataptr = ival.data_ptr<long>();
    if (ival_dataptr[0] != 0) return false;
  }

  // check if reshape is the output of the graph
  auto reshape_pattern = map.at("reshape");
  auto reshape_node = match.values_map.at(reshape_pattern);
  auto uses = reshape_node->uses();
  for (auto use : uses) {
    auto user = use.user;
    if (is_kind(user, "prim::Return")) return false;
  }

  return true;
}

// from:
// x->shape->gather->unsqueeze->concat
// |                              |
// gap--------------------------reshape
//
// to:
// x->gap->flatten
void FlattenClsHead(std::shared_ptr<Graph>& graph) {
  std::string pattern = R"IR(
      graph(%x, %cat0, %cat1):
        %gap = onnx::GlobalAveragePool(%x)
        %cat = onnx::Concat[axis=0](%cat0, %cat1)
        %reshape = onnx::Reshape(%gap, %cat)
        return (%reshape)
  )IR";

  std::string replacement = R"IR(
      graph(%x, %cat0, %cat1):
        %gap = onnx::GlobalAveragePool(%x)
        %flatten = onnx::Flatten(%gap)
        return (%flatten)
  )IR";

  torch::jit::SubgraphRewriter subgraph_rewriter;
  subgraph_rewriter.RegisterRewritePattern(pattern, replacement);
  subgraph_rewriter.runOnGraph(graph, matchClsHead);

  torch::jit::EliminateDeadCode(
      graph->block(), true,
      torch::jit::DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

}  // namespace torch_jit
}  // namespace mmdeploy
