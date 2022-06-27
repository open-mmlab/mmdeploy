#include "Pass/Common/Simplify.h"

namespace ir {
namespace simplify {

#define RESULT_POP_AS(R) Polynomial R = pop();
#define RESULT_PUSH(R) push(R);

#define SINGLE_NODE(expr) (Polynomial(Monomial(expr)))

ExprPtr NumericSimplifier::simplifyExpr(Expr* node) {
  NumericSimplifier simplifier;
  simplifier.visit(node);
  // std::cout << ">> Pop Result: ";
  // api::dump_expr(simplifier.results_.top().asExpr(), std::cout);
  ELENA_ASSERT_EQ(
      simplifier.results_.size(), 1,
      "The size of the stack should be 1. Check the nodes visited!");

  return simplifier.results_.top().asExpr();
}

void NumericSimplifier::visit(IterVar* node) {
  RESULT_PUSH(SINGLE_NODE(node->shared_from_this()));
}

void NumericSimplifier::visit(ScalarVar* node) {
  if (node->tensor && node->indices) {
    CAST_TO(indices, Array<Expr>, ExprSimplifier::simplifyNode(node->indices));
    auto var =
        std::make_shared<ScalarVar>(node->tensor, indices, node->get_name());
    RESULT_PUSH(SINGLE_NODE(var));
    return;
  }
  RESULT_PUSH(SINGLE_NODE(node->shared_from_this()));
}

void NumericSimplifier::visit(TensorVar* node) {
  RESULT_PUSH(SINGLE_NODE(node->shared_from_this()));
}

void NumericSimplifier::visit(Unary* node) {
  UnaryType op = node->operation_type;
  visit(node->operand.get());
  RESULT_POP_AS(opd);

  switch (op) {
    case UnaryType::Negate:
      RESULT_PUSH(-opd);
      return;
    case UnaryType::Ceil:
      if (opd.isMonomial()) {
        auto copd = opd.asMonomial();
        if (copd.isConstant()) {
          RESULT_PUSH(SINGLE_NODE(
              PatternSimplifier::eval(copd.asExpr(), UnaryType::Ceil)));
          return;
        }
      }

    default:
      break;
  }

  if (op == UnaryType::Negate) {
    RESULT_PUSH(-opd);
    return;
  }

  RESULT_PUSH(SINGLE_NODE(std::make_shared<Unary>(opd.asExpr(), op)));
}

void NumericSimplifier::visit(Binary* node) {
  BinaryType op = node->operation_type;
  visit(node->lhs.get());
  visit(node->rhs.get());
  RESULT_POP_AS(rhs);
  RESULT_POP_AS(lhs);

  switch (op) {
    case BinaryType::Add:
      RESULT_PUSH(lhs + rhs);
      return;
    case BinaryType::Sub:
      RESULT_PUSH(lhs - rhs);
      return;
    case BinaryType::Mul:
      RESULT_PUSH(lhs * rhs)
      return;
    case BinaryType::Div:
      if (lhs.isMonomial() && rhs.isMonomial()) {
        auto clhs = lhs.asMonomial();
        auto crhs = rhs.asMonomial();
        if (clhs.isConstant() && crhs.isConstant()) {
          RESULT_PUSH(SINGLE_NODE(PatternSimplifier::eval(
              clhs.asExpr(), crhs.asExpr(), BinaryType::Div)));
          return;
        }
      }
    default:
      break;
  }

  auto lexpr = lhs.asExpr();
  auto rexpr = rhs.asExpr();
  RESULT_PUSH(SINGLE_NODE(std::make_shared<Binary>(lexpr, rexpr, op)));
}

void NumericSimplifier::visit(Cast* node) {
  // To simplify `Cast`, simplify the inner expression first. If it is an
  // constant, cast and push it to the stack. Otherwise, build a new Cast node
  // and push it.

  ExprPtr expr = api::simplify(node->expr_);
  ScalarType dst_type = node->get_dtype();

  RESULT_PUSH(SINGLE_NODE(PatternSimplifier::eval(expr, dst_type)));
}

void NumericSimplifier::visit(Select* node) {
  // To simplify `Select`, simplify condtion first. If it is a constant, choose
  // and simplify the branch. Otherwise, simplify each branch and build a new
  // Select then push it.

  ExprPtr cond = api::simplify(node->cond);
  ELENA_ASSERT_EQ(cond->get_dtype(), ScalarType::Boolean,
                  "The conditon of Select should be boolean type");

  if (cond->get_type() == IRNodeType::Const) {
    auto con = ptr_cast<Const<bool>>(cond);
    RESULT_PUSH(SINGLE_NODE(con->get_value() ? api::simplify(node->tBranch)
                                             : api::simplify(node->fBranch)));
    return;
  }

  ExprPtr tBranch = api::simplify(node->tBranch);
  ExprPtr fBranch = api::simplify(node->fBranch);
  RESULT_PUSH(SINGLE_NODE(std::make_shared<Select>(cond, tBranch, fBranch)));
}

void NumericSimplifier::visit(Call* node) {
  // To simplify `Call`, simplify each arguments and build a new Call node.

  std::vector<ExprPtr> args;
  CallFunction func = node->func;
  ScalarType dtype = node->get_dtype();
  for (auto& e : node->args->element) {
    args.push_back(api::simplify(e));
  }

  auto args_ = std::make_shared<Array<Expr>>(args);
  RESULT_PUSH(SINGLE_NODE(std::make_shared<Call>(func, args_, dtype)));
}

void NumericSimplifier::push(Polynomial poly) {
  // std::cout << ">> Push (" << results_.size() + 1 << "): ";
  // api::dump_expr(poly.as_expr(), std::cout);
  results_.push(poly);
}

Polynomial NumericSimplifier::pop() {
  auto p = results_.top();
  results_.pop();
  // std::cout << ">> Pop (" << results_.size() << "):  ";
  // api::dump_expr(p.as_expr(), std::cout);
  return p;
}

}  // namespace simplify
}  // namespace ir
