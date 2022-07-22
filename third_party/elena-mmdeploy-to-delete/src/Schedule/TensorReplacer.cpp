#include "Schedule/TensorReplacer.h"

TensorReplacer::TensorReplacer(const ir::TensorVarMap<ir::TensorVarPtr>& tmap)
    : replace_map(tmap) {}

ir::NodePtr TensorReplacer::MutateReplace(ir::NodePtr node) {
  mutate(node);
  return node;
}

ir::NodePtr TensorReplacer::visit(ir::TensorVar* tensor_ptr) {
  auto it = replace_map.find(tensor_ptr->shared_from_this());
  if (it != replace_map.end()) {
    return MutatorBase::visit(it->second.get());
  } else {
    return MutatorBase::visit(tensor_ptr);
  }
}
