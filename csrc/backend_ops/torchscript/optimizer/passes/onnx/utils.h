#ifndef _PASSES_ONNX_UTILS_H_
#define _PASSES_ONNX_UTILS_H_

#include <torch/script.h>

namespace mmdeploy {
namespace torch_jit {
using c10::Symbol;
using torch::jit::Node;

inline bool is_kind(const Node* node, const Symbol& symbol) { return node->kind() == symbol; }

inline bool is_kind(const Node* node, const char* symbol_name) {
  return is_kind(node, Symbol::fromQualString(symbol_name));
}

}  // namespace torch_jit
}  // namespace mmdeploy

#endif
