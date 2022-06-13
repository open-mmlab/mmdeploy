// Copyright (c) OpenMMLab. All rights reserved.
#ifndef _MERGE_SHAPE_CONCATE_H_
#define _MERGE_SHAPE_CONCATE_H_

#include <torch/script.h>
namespace mmdeploy {
namespace torch_jit {
using torch::jit::Graph;

void MergeShapeConcate(const std::shared_ptr<Graph>& graph);
}  // namespace torch_jit
}  // namespace mmdeploy

#endif
