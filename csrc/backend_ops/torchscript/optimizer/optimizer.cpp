#include "optimizer.h"

#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>

#if TORCH_VERSION_MINOR >= 9
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#endif

namespace mmdeploy {
Module optimize_for_torchscript(const Module& model) {
  auto frozen_model = freeze_module(model);
  auto graph = frozen_model.get_method("forward").graph();
  OptimizeFrozenGraph(graph, true);

#if TORCH_VERSION_MINOR >= 9
  FuseFrozenConvAddRelu(graph);
  ConvertFrozenOpsToMKLDNN(graph);
  FrozenLinearTranspose(graph);
#endif

  // TODO: add more custom passes

  return frozen_model;
}

Module optimize_for_onnx(const Module& model) {
  auto frozen_model = freeze_module(model, {"training"});
  auto graph = frozen_model.get_method("forward").graph();
  OptimizeFrozenGraph(graph, true);

#if TORCH_VERSION_MINOR >= 9
  FuseFrozenConvAddRelu(graph);
  ConvertFrozenOpsToMKLDNN(graph);
  FrozenLinearTranspose(graph);
#endif

  // TODO: add more custom passes

  return frozen_model;
}

// TODO: add optimizer for other backend/onnx

}  // namespace mmdeploy
