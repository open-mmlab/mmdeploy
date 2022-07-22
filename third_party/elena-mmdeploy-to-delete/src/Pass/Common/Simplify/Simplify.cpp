#include "Pass/Common/Simplify.h"

#include "api.h"

namespace ir {
namespace simplify {

NodePtr ExprSimplifier::simplifyNode(Node* node) {
  if (!node) return nullptr;

  // The label should not be a subclass of Expr in X-macro
  if (::ir::cast_check::is_compatible<Expr>(node->get_type()) &&
      node->get_type() != IRNodeType::Label &&
      node->get_type() != IRNodeType::Array)
    return ExprSimplifier::simplifyExpr(static_cast<Expr*>(node));

  ExprSimplifier simplifier;
  return simplifier.visit(node);
}

NodePtr ExprSimplifier::simplifyNode(NodePtr node) {
  return ExprSimplifier::simplifyNode(node.get());
}

ExprPtr ExprSimplifier::simplifyExpr(Expr* node) {
  if (!node) return nullptr;

  ExprPtr expr = ptr_cast<Expr>(PreSimplifier::simplifyNode(node));

  if (expr->get_dtype() == ScalarType::Boolean)
    expr = BoolSimplifier::simplifyNode(expr.get());
  else
    expr = NumericSimplifier::simplifyExpr(expr.get());

  expr = ptr_cast<Expr>(PostSimplifier::simplifyNode(expr.get()));
  return expr;
}

ExprPtr ExprSimplifier::simplifyExpr(ExprPtr node) {
  // std::cout << "[=================== BEGIN ===================]\n";
  // api::dump_expr(node, std::cout);

  node = ExprSimplifier::simplifyExpr(node.get());

  // api::dump_expr(node, std::cout);
  // std::cout << "[=================== END ===================]\n";

  return node;
}

NodePtr ExprSimplifier::visit(For* node) {
  auto init = simplifyExpr(node->init);
  auto extent = simplifyExpr(node->extent);
  CAST_TO(body, Stmt, visit(node->body.get()));
  return std::make_shared<For>(node->it, init, extent, node->for_type, body);
}

NodePtr ExprSimplifier::visit(Block* node) {
  CAST_TO(head, Stmt, visit(node->head.get()));
  if (node->tail) {
    CAST_TO(tail, Stmt, visit(node->tail.get()));
    return std::make_shared<Block>(head, tail);
  }
  return std::make_shared<Block>(head);
}

NodePtr ExprSimplifier::visit(Provide* node) {
  auto value = simplifyExpr(node->value);
  CAST_TO(index, Array<Expr>, visit(node->index.get()));
  return std::make_shared<Provide>(node->var, value, index);
}

NodePtr ExprSimplifier::visit(Store* node) {
  auto value = simplifyExpr(node->value);
  CAST_TO(index, Array<Expr>, visit(node->index.get()));
  return std::make_shared<Store>(node->var, value, index);
}

NodePtr ExprSimplifier::visit(Range* node) {
  auto init = simplifyExpr(node->init);
  auto extent = simplifyExpr(node->extent);
  if (node->stride) {
    auto stride = simplifyExpr(node->stride);
    return std::make_shared<Range>(init, extent, stride);
  }
  return std::make_shared<Range>(init, extent);
}

NodePtr ExprSimplifier::visit(Allocate* node) {
  CAST_TO(var, Var, visit(node->var.get()));
  CAST_TO(bound, Array<Range>, visit(node->bound.get()));
  CAST_TO(body, Stmt, visit(node->body.get()));
  return std::make_shared<Allocate>(var, bound, body, node->is_output);
}

NodePtr ExprSimplifier::visit(IfThenElse* node) {
  auto cond = simplifyExpr(node->condition);
  CAST_TO(tb, Stmt, visit(node->then_case.get()));
  if (node->else_case) {
    CAST_TO(eb, Stmt, visit(node->else_case.get()));
    return std::make_shared<IfThenElse>(cond, tb, eb);
  }
  return std::make_shared<IfThenElse>(cond, tb, nullptr);
}

NodePtr ExprSimplifier::visit(Let* node) {
  auto value = simplifyExpr(node->value);
  CAST_TO(body, Stmt, visit(node->body.get()));
  return std::make_shared<Let>(node->var, value, body);
}

NodePtr ExprSimplifier::visit(Attr* node) {
  NodePtr value = simplifyNode(node->value);  // visit(node->value.get());
  CAST_TO(body, Stmt, visit(node->body.get()));
  return std::make_shared<Attr>(node->node, node->key, value, body);
}

}  // namespace simplify
}  // namespace ir

namespace api {

using namespace ir;            // NOLINT
using namespace ir::simplify;  // NOLINT

ExprPtr simplify(ExprPtr expr) {
  expr = ptr_cast<Expr>(ExprSimplifier::simplifyExpr(expr));

  return expr;
}

StmtPtr simplify(StmtPtr node) {
  // std::cout << "[=================== BEGIN STMT ===================]\n";
  // api::dump_stmt(node, std::cout);

  node = ptr_cast<ir::Stmt>(ExprSimplifier::simplifyNode(node));

  // api::dump_stmt(node, std::cout);
  // std::cout << "[=================== END STMT ===================]\n";

  return node;
}

}  // namespace api
