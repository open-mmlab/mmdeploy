#ifndef _PASSES_ONNX_UTILS_H_
#define _PASSES_ONNX_UTILS_H_

#include <torch/script.h>

namespace mmdeploy {
namespace torch_jit {
using c10::Symbol;
using torch::jit::Node;
bool is_kind(const Node* node, const Symbol& symbol) {
  return strcmp(node->kind().toQualString(), symbol.toQualString()) == 0;
}
}  // namespace torch_jit
}  // namespace mmdeploy

#endif
