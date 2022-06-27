#ifndef ELENA_INCLUDE_IR_EXPRVISITOR_H_
#define ELENA_INCLUDE_IR_EXPRVISITOR_H_

#include <unordered_set>

#include "VisitorBase.h"

/**
 * @brief This visitor is used to find the operand tensor given an expr
 * @author hanruobing
 */

class ExprVisitor : public VisitorBase<ExprVisitor> {
 public:
  ExprVisitor();

  ir::ArrayPtr<ir::TensorVar> tensorInExpr(ir::NodePtr node);

  std::unordered_set<ir::TensorVarPtr> visit_set;

  void visit(ir::TensorVar* node);
  void visit(ir::IterVar* node);
  using VisitorBase::visit;
};

#endif  // ELENA_INCLUDE_IR_EXPRVISITOR_H_
