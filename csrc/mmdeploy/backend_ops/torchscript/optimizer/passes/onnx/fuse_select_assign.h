// Copyright (c) OpenMMLab. All rights reserved.
#ifndef _FUSE_SELECT_ASSIGN_H_
#define _FUSE_SELECT_ASSIGN_H_

#include <torch/script.h>
namespace mmdeploy {
namespace torch_jit {
using torch::Tensor;
using torch::jit::Graph;

// this pass is used to fuse y[x>thres] = z[x>thres]
void FuseSelectAssign(std::shared_ptr<Graph>& graph,
                      std::unordered_map<std::string, Tensor>& params);
}  // namespace torch_jit
}  // namespace mmdeploy

#endif
