// Copyright (c) OpenMMLab. All rights reserved.
#ifndef _FLATTEN_CLS_HEAD_H_
#define _FLATTEN_CLS_HEAD_H_

#include <torch/script.h>
namespace mmdeploy {
namespace torch_jit {
using torch::jit::Graph;

void FlattenClsHead(std::shared_ptr<Graph>& graph);
}  // namespace torch_jit
}  // namespace mmdeploy

#endif
