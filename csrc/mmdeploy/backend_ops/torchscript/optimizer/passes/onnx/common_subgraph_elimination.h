// Copyright (c) OpenMMLab. All rights reserved.
#ifndef _COMMON_SUBGRAPH_ELIMINATION_H_
#define _COMMON_SUBGRAPH_ELIMINATION_H_

#include <torch/script.h>
namespace mmdeploy {
namespace torch_jit {
using torch::Tensor;
using torch::jit::Graph;

// This pass is used eliminate the common subgraph.
// There are two main difference between the one in torch/csrc/jit/pass
// 1. AliasDb is not needed in ONNX model
// 2. params might also participated in the elimination
void CommonSubgraphElimination(std::shared_ptr<Graph>& graph,
                               std::unordered_map<std::string, Tensor>& params);
}  // namespace torch_jit
}  // namespace mmdeploy

#endif
