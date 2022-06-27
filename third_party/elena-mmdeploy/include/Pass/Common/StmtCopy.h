#ifndef ELENA_INCLUDE_PASS_COMMON_STMTCOPY_H_
#define ELENA_INCLUDE_PASS_COMMON_STMTCOPY_H_

#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IR/Expr.h"
#include "IR/MutatorBase.h"
#include "IR/Type.h"

class StmtCopy : public MutatorBase<StmtCopy> {
 public:
  using MutatorBase::visit;
  ir::NodePtr visit(ir::Provide *node) {
    auto var_ptr = ir::ptr_cast<ir::TensorVar>(visit(node->var.get()));
    auto value_ptr = ir::ptr_cast<ir::Expr>(visit(node->value.get()));
    auto index_ptr =
        ir::ptr_cast<ir::Array<ir::Expr>>(visit(node->index.get()));
    return std::make_shared<ir::Provide>(var_ptr, value_ptr, index_ptr);
  }

  ir::NodePtr visit(ir::Store *node) {
    auto var_ptr = ir::ptr_cast<ir::TensorVar>(visit(node->var.get()));
    auto value_ptr = ir::ptr_cast<ir::Expr>(visit(node->value.get()));
    auto index_ptr =
        ir::ptr_cast<ir::Array<ir::Expr>>(visit(node->index.get()));
    return std::make_shared<ir::Store>(var_ptr, value_ptr, index_ptr);
  }

  ir::NodePtr visit(ir::Let *node) {
    auto var_ptr = ir::ptr_cast<ir::Var>(visit(node->var.get()));
    auto value_ptr = ir::ptr_cast<ir::Expr>(visit(node->value.get()));
    auto body_ptr = ir::ptr_cast<ir::Stmt>(visit(node->body.get()));
    return std::make_shared<ir::Let>(var_ptr, value_ptr, body_ptr);
  }

  ir::NodePtr visit(ir::IterVar *node) {
    auto iter = std::make_shared<ir::IterVar>(node->range, node->get_name(),
                                              node->is_reduce);
    iter->iter_type = node->iter_type;
    return iter;
  }

  ir::NodePtr visit(ir::ScalarVar *node) {
    if (!node->is_placeholder()) {
      auto tensor_ptr = ir::ptr_cast<ir::TensorVar>(visit(node->tensor.get()));
      auto indices_ptr =
          ir::ptr_cast<ir::Array<ir::Expr>>(visit(node->indices.get()));
      return std::make_shared<ir::ScalarVar>(tensor_ptr, indices_ptr,
                                             node->get_name());
    } else {
      return node->shared_from_this();
    }
  }

  ir::NodePtr visit(ir::Select *node) {
    auto cond = ir::ptr_cast<ir::Expr>(visit(node->cond.get()));
    auto tBranch = ir::ptr_cast<ir::Expr>(visit(node->tBranch.get()));
    auto fBranch = ir::ptr_cast<ir::Expr>(visit(node->fBranch.get()));
    return std::make_shared<ir::Select>(cond, tBranch, fBranch);
  }

  ir::NodePtr visit(ir::IfThenElse *node) {
    auto condition_ptr = ir::ptr_cast<ir::Expr>(visit(node->condition.get()));
    auto then_ptr = ir::ptr_cast<ir::Stmt>(visit(node->then_case.get()));
    ir::StmtPtr else_ptr = nullptr;
    if (node->else_case) {
      else_ptr = ir::ptr_cast<ir::Stmt>(visit(node->else_case.get()));
    }
    return std::make_shared<ir::IfThenElse>(condition_ptr, then_ptr, else_ptr);
  }

  ir::NodePtr visit(ir::Block *node) {
    auto head_ptr = ir::ptr_cast<ir::Stmt>(visit(node->head.get()));
    auto tail_ptr = ir::ptr_cast<ir::Stmt>(visit(node->tail.get()));
    return std::make_shared<ir::Block>(head_ptr, tail_ptr);
  }

  ir::NodePtr visit(ir::Logical *node) {
    auto lhs_ptr = ir::ptr_cast<ir::Expr>(visit(node->lhs.get()));
    auto rhs_ptr = ir::ptr_cast<ir::Expr>(visit(node->rhs.get()));
    return std::make_shared<ir::Logical>(lhs_ptr, rhs_ptr,
                                         node->operation_type);
  }

  ir::NodePtr visit(ir::Binary *node) {
    auto lhs = ir::ptr_cast<ir::Expr>(visit(node->lhs.get()));
    auto rhs = ir::ptr_cast<ir::Expr>(visit(node->rhs.get()));
    return std::make_shared<ir::Binary>(lhs, rhs, node->operation_type);
  }

  ir::NodePtr visit(ir::Unary *node) {
    auto operand = ir::ptr_cast<ir::Expr>(visit(node->operand.get()));
    return std::make_shared<ir::Unary>(operand, node->operation_type);
  }

  template <typename T>
  ir::NodePtr visit(ir::Array<T> *array_ptr) {
    std::vector<std::shared_ptr<T>> element;
    // recursively visit all elements in the array.
    for (auto &a : array_ptr->element) {
      element.push_back(ir::ptr_cast<T>(visit(a.get())));
    }
    return std::make_shared<ir::Array<T>>(element);
  }

  ir::NodePtr stmt_copy(ir::NodePtr node) { return visit(node.get()); }
};

#endif  // ELENA_INCLUDE_PASS_COMMON_STMTCOPY_H_
