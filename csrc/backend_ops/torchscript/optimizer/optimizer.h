#include <torch/script.h>

namespace mmdeploy {
using torch::jit::script::Module;

Module optimize_for_torchscript(const Module &model);
}  // namespace mmdeploy
