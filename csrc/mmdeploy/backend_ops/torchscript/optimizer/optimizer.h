// Copyright (c) OpenMMLab. All rights reserved.
#include <torch/script.h>

namespace mmdeploy {
using torch::jit::script::Module;

Module optimize_for_torchscript(const Module &model);

Module optimize_for_onnx(const Module &model);
}  // namespace mmdeploy
