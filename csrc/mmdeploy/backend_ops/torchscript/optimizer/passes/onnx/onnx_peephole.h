// Copyright (c) OpenMMLab. All rights reserved.
#ifndef _ONNX_PEEPHOLE_H_
#define _ONNX_PEEPHOLE_H_

#include <torch/script.h>
namespace mmdeploy {
namespace torch_jit {
using torch::jit::Graph;

void ONNXPeephole(const std::shared_ptr<Graph>& graph);

}  // namespace torch_jit
}  // namespace mmdeploy

#endif
