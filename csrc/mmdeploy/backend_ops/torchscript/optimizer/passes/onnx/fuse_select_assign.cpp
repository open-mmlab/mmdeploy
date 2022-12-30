#include "fuse_select_assign.h"

#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "../../ir/subgraph_matcher.h"
#include "common_subgraph_elimination.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace mmdeploy {
namespace torch_jit {

using c10::Symbol;
using torch::jit::Block;
using torch::jit::IValue;
using torch::jit::Node;

bool RemoveBoolCast(Node* node) {
  auto bottom_node = node->input()->node();
  if (bottom_node->kind() != Symbol::onnx("Greater") &&
      bottom_node->kind() != Symbol::onnx("Less")) {
    return false;
  }
  node->output()->replaceAllUsesWith(bottom_node->output());
  return true;
}

bool FuseSelectAssign(Node* node, std::unordered_map<std::string, Tensor>& params,
                      std::unordered_map<std::string, Value*>& vmap, SubgraphMatcher& matcher) {
  auto values_map = matcher.values_map();

  auto cmp1 = values_map[vmap["cmp_1"]]->node();
  auto cmp2 = values_map[vmap["cmp_2"]]->node();
  if (cmp1 != cmp2) {
    // cmp_1 == cmp_2, cmp in (Great, Less)
    if (cmp1->kind() != cmp2->kind()) return false;
    if (!(cmp1->kind() == Symbol::onnx("Greater") || cmp1->kind() == Symbol::onnx("Less")))
      return false;

    // check threshold
    Node* cmps[] = {cmp1, cmp2};
    float thres = 0.0f;
    Node* x = nullptr;
    for (int i = 0; i < 2; ++i) {
      auto cmp = cmps[i];
      auto threshold = cmp->inputs()[1]->node();
      if (threshold->kind() != Symbol::onnx("Constant")) return false;
      auto thres_val = threshold->t(Symbol::attr("value"));
      if (i == 0) {
        thres = thres_val.data_ptr<float>()[0];
        x = cmp->inputs()[0]->node();
      } else {
        float tmp_val = thres_val.data_ptr<float>()[0];
        if (fabs(thres - tmp_val) > 1e-10) {
          return false;
        }
        if (x != cmp->inputs()[0]->node()) {
          return false;
        }
      }
    }
  }

  {
    // check shape of reshape
    Node* shape = values_map[vmap["reshape_1_shape"]]->node();
    auto shape_val = shape->t(Symbol::attr("value"));
    if (shape_val.dim() != 1) return false;
    if (shape_val.data_ptr<int64_t>()[0] != -1) return false;
  }

  {
    // check transpose
    Node* trans[] = {values_map[vmap["trans_1"]]->node(), values_map[vmap["trans_2"]]->node()};
    for (auto tran : trans) {
      auto tran_perm = tran->is(Symbol::attr("perm"));
      if (tran_perm.size() != 2) return false;
      if (tran_perm[0] != 1 || tran_perm[1] != 0) return false;
    }
  }

  {
    // check gather indice
    Node* gather_inds = values_map[vmap["gather_inds_2"]]->node();
    auto inds_val = gather_inds->t(Symbol::attr("value"));
    if (inds_val.dim() != 0) return false;
    if (inds_val.data_ptr<int64_t>()[0] != 0) return false;
  }

  {
    // check slice start
    Node* slice = values_map[vmap["slice_2"]]->node();
    auto start_name = slice->inputs()[1]->debugName();
    auto start_val = params[start_name];
    if (start_val.dim() != 1) return false;
    if (start_val.data_ptr<int64_t>()[0] != 0) return false;
  }

  // create new node
  auto graph = node->owningGraph();
  auto z = values_map[vmap["z"]];
  auto y = values_map[vmap["y"]];
  auto where_node = graph->create(Symbol::onnx("Where"), {cmp1->output(), z, y});
  where_node->insertBefore(node);
  where_node->output()->copyMetadata(node->output());
  node->output()->replaceAllUsesWith(where_node->output());
  return true;
}

void FuseSelectAssign(Block* block, std::unordered_map<std::string, Tensor>& params,
                      std::unordered_map<std::string, Value*>& vmap, SubgraphMatcher& matcher) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      FuseSelectAssign(block, params, vmap, matcher);
    }

    if (node->kind() == Symbol::onnx("Cast") && node->i(Symbol::attr("to")) == 9) {
      RemoveBoolCast(node);
    } else if (matcher.matchesSubgraphFromAnchorNode(node)) {
      FuseSelectAssign(node, params, vmap, matcher);
    }
  }
}

void FuseSelectAssign(std::shared_ptr<Graph>& graph,
                      std::unordered_map<std::string, Tensor>& params) {
  // cse before search
  CommonSubgraphElimination(graph, params);

  std::string pattern_str = R"IR(
      graph(%y, %z, %cmp_1, %cmp_2, %start, %axes, %shape_2):
        %nz_1 = onnx::NonZero(%cmp_1)
        %trans_1 = onnx::Transpose(%nz_1)
        %gather_1 = onnx::GatherND(%z, %trans_1)
        %reshape_1_shape = onnx::Constant()
        %reshape_1 = onnx::Reshape(%gather_1, %reshape_1_shape)
        %expand_2 = onnx::Expand(%cmp_2, %shape_2)
        %nz_2 = onnx::NonZero(%expand_2)
        %trans_2 = onnx::Transpose(%nz_2)
        %trans_shape_2 = onnx::Shape(%trans_2)
        %gather_inds_2 = onnx::Constant()
        %gather_2 = onnx::Gather(%trans_shape_2, %gather_inds_2)
        %unsqueeze_2 = onnx::Unsqueeze(%gather_2)
        %slice_2 = onnx::Slice(%reshape_1, %start, %unsqueeze_2, %axes)
        %scatter_2 = onnx::ScatterND(%y, %trans_2, %slice_2)
        return (%scatter_2)
  )IR";

  Graph pattern;
  std::unordered_map<std::string, Value*> vmap;
  torch::jit::parseIR(pattern_str, &pattern, vmap);

  SubgraphMatcher matcher(pattern, MatchAttribute::NO_MATCH);
  FuseSelectAssign(graph->block(), params, vmap, matcher);
  torch::jit::EliminateDeadCode(
      graph->block(), true,
      torch::jit::DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}
}  // namespace torch_jit
}  // namespace mmdeploy
