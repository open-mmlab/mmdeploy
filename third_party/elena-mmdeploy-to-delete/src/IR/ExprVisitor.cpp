#include "IR/ExprVisitor.h"

using ir::IRNodeType;
using ir::Node;

ExprVisitor::ExprVisitor() {}

ir::ArrayPtr<ir::TensorVar> ExprVisitor::tensorInExpr(ir::NodePtr node) {
  visit(node);
  ir::ArrayPtr<ir::TensorVar> ret =
      std::make_shared<ir::Array<ir::TensorVar>>();
  for (auto i : visit_set) {
    ret->element.push_back(i);
  }
  return ret;
}

void ExprVisitor::visit(ir::TensorVar* tensor_ptr) {
  visit_set.insert(tensor_ptr->shared_from_this());
  visit(tensor_ptr->shape.get());
}

void ExprVisitor::visit(ir::IterVar* iter_ptr) {
  visit(iter_ptr->range->init.get());
  visit(iter_ptr->range->extent.get());
}
