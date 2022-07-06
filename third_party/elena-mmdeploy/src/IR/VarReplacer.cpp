#include "IR/VarReplacer.h"

using ir::IRNodeType;
using ir::Node;

VarReplacer::VarReplacer(
    const std::unordered_map<ir::NodePtr, ir::NodePtr>& vmap)
    : replace_map(vmap) {}

ir::NodePtr VarReplacer::visit(ir::ScalarVar* scalar_var_ptr) {
  auto it = replace_map[scalar_var_ptr->shared_from_this()];
  if (it) {
    return it;
  } else {
    std::vector<ir::ExprPtr> indices;
    if (scalar_var_ptr->tensor) {
      mutate(scalar_var_ptr->tensor);
    }
    if (scalar_var_ptr->indices) {
      mutate(scalar_var_ptr->indices);
    }
    return scalar_var_ptr->shared_from_this();
  }
}

ir::NodePtr VarReplacer::visit(ir::IterVar* iter_var_ptr) {
  auto it = replace_map[iter_var_ptr->shared_from_this()];
  if (it) {
    return it;
  } else {
    return iter_var_ptr->shared_from_this();
  }
}

ir::NodePtr VarReplacer::mutateReplace(ir::NodePtr node) {
  mutate(node);
  return node;
}
