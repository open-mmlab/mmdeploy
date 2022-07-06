#ifndef ELENA_INCLUDE_IR_INLINEMUTATE_H_
#define ELENA_INCLUDE_IR_INLINEMUTATE_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "MutatorBase.h"
#include "Stmt.h"
#include "VisitorBase.h"
#include "api.h"

class ExprReConstructor final : public VisitorBase<ExprReConstructor> {
 public:
  using VisitorBase<ExprReConstructor>::visit;

  template <typename T>
  void visit(ir::Const<T>* const_ptr) {
    result = std::make_shared<Const<T>>(*const_ptr);
  }

  void visit(ir::Select* select_ptr) {
    visit(select_ptr->cond);
    auto cond = result;
    visit(select_ptr->fBranch);
    auto fBranch = result;
    visit(select_ptr->tBranch);
    auto tBranch = result;
    result = ir::ExprPtr(new ir::Select(cond, tBranch, fBranch));
  }

  void visit(ir::IfThenElse* if_then_else_ptr) {
    visit(if_then_else_ptr->condition);
    auto cond = result;
    visit(if_then_else_ptr->then_case);
    auto then_case = result;
    visit(if_then_else_ptr->else_case);
    auto else_case = result;
    result = api::if_then_else(cond, then_case, else_case);
  }

  void visit(ir::Logical* logical_ptr) {
    visit(logical_ptr->lhs);
    auto lhs = result;
    visit(logical_ptr->rhs);
    auto rhs = result;
    result =
        ir::ExprPtr(new ir::Logical(lhs, rhs, logical_ptr->operation_type));
  }

  void visit(ir::Cast* cast_ptr) {
    visit(cast_ptr->expr_);
    auto expr_ = result;
    result = ir::ExprPtr(new ir::Cast(expr_, cast_ptr->get_dtype()));
  }

  void visit(ir::Binary* binary_ptr) {
    visit(binary_ptr->lhs.get());
    ExprPtr lhs = result;
    visit(binary_ptr->rhs.get());
    ExprPtr rhs = result;

    result = ExprPtr(new ir::Binary(lhs, rhs, binary_ptr->operation_type));
  }

  void visit(ir::Unary* unary_ptr) {
    visit(unary_ptr->operand.get());
    result = ExprPtr(new ir::Unary(result, unary_ptr->operation_type));
  }

  void visit(ir::ScalarVar* scalar_ptr) {
    if (!scalar_ptr->tensor) {
      result = scalar_ptr->shared_from_this();
      return;
    }
    std::vector<ir::ExprPtr> indices;
    for (auto& iter : scalar_ptr->indices->element) {
      visit(iter.get());
      indices.push_back(result);
    }
    result = ir::ScalarVarPtr(new ir::ScalarVar(
        scalar_ptr->tensor, std::make_shared<ir::Array<ir::Expr>>(indices),
        scalar_ptr->get_name()));
  }

  void visit(ir::IterVar* iter_ptr) { result = iter_ptr->shared_from_this(); }

  ExprPtr getConstructedExpr(const ir::NodePtr& node) {
    visit(node.get());
    return result;
  }

 private:
  ExprPtr result;
};

class ItervarMutate : public MutatorBase<ItervarMutate> {
 public:
  explicit ItervarMutate(std::unordered_map<ir::IterVarPtr, ir::ExprPtr> imap);
  ir::NodePtr mutateItervar(ir::NodePtr node);
  ir::NodePtr visit(ir::IterVar* node);

  using MutatorBase::visit;

 private:
  std::unordered_map<ir::IterVarPtr, ir::ExprPtr> iter_map;
};

class InlineMutate : public MutatorBase<InlineMutate> {
 public:
  InlineMutate(ir::OpPtr op, ir::ArrayPtr<ir::IterVar> args, ir::ExprPtr expr);
  ir::NodePtr mutateInline(ir::NodePtr node);
  ir::NodePtr visit(ir::ScalarVar* node);

  using MutatorBase::visit;

 private:
  ir::OpPtr iop;
  ir::ArrayPtr<ir::IterVar> iargs;
  ir::ExprPtr iexpr;
};

ir::ExprPtr inlineExpr(ir::ExprPtr cexpr, ir::OpPtr op,
                       ir::ArrayPtr<ir::IterVar> args, ir::ExprPtr expr);

#endif  // ELENA_INCLUDE_IR_INLINEMUTATE_H_
