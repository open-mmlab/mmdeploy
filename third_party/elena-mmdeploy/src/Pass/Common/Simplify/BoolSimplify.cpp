#include "Pass/Common/Simplify.h"

namespace {

template <typename T>
ExprPtr makeConst(T value, ScalarType dtype) {
  return std::make_shared<Const<T>>(value, dtype);
}

#define CONST_TRUE (std::make_shared<Const<bool>>(true, ScalarType::Boolean))
#define CONST_FALSE (std::make_shared<Const<bool>>(false, ScalarType::Boolean))

}  // namespace

namespace ir {
namespace simplify {

ExprPtr BoolSimplifier::simplifyNode(Expr* node) {
  BoolSimplifier simplifier;
  return ir::ptr_cast<Expr>(simplifier.visit(node));
}

ExprPtr BoolSimplifier::visit(Unary* node) {
  if (node->get_dtype() != ScalarType::Boolean) {
    visit(node->operand.get());
    return node->shared_from_this();
  }
  ELENA_ASSERT_EQ(node->get_dtype(), ScalarType::Boolean,
                  "Unary in BoolSimplifier::visit should have type Boolean.");

  NodePtr opd = visit(node->operand.get());
  UnaryType op = node->operation_type;

  // ~(Const b) -> Const ~b
  if (op == UnaryType::Negate && opd->get_type() == IRNodeType::Const) {
    ConstPtr<bool> opd_bool = ptr_cast<Const<bool>>(opd);
    ELENA_ASSERT(opd_bool, "operand of Unary Negate should have type bool");

    return std::make_shared<Const<bool>>(!opd_bool->get_value(),
                                         ScalarType::Boolean);
  }

  return node->shared_from_this();
}

ExprPtr BoolSimplifier::visit(Binary* node) {
  if (node->get_dtype() != ScalarType::Boolean) {
    visit(node->lhs.get());
    visit(node->rhs.get());
    return node->shared_from_this();
  }
  ELENA_ASSERT_EQ(node->get_dtype(), ScalarType::Boolean,
                  "Binary in BoolSimplifier::visit should have type Boolean.");

  CAST_TO(lhs, Expr, visit(node->lhs.get()));
  CAST_TO(rhs, Expr, visit(node->rhs.get()));
  BinaryType op = node->operation_type;

  if (op == BinaryType::And) {
    // False && _ -> False
    if (EQ(lhs, CONST_FALSE)) return CONST_FALSE;
    // _ && False -> False
    if (EQ(rhs, CONST_FALSE)) return CONST_FALSE;
    // True && rhs -> rhs
    if (EQ(lhs, CONST_TRUE)) return rhs;
    // lhs && True -> lhs
    if (EQ(rhs, CONST_TRUE)) return lhs;
  }

  if (op == BinaryType::Or) {
    // True || _ -> True
    if (EQ(lhs, CONST_TRUE)) return CONST_TRUE;
    // _ || True -> True
    if (EQ(rhs, CONST_TRUE)) return CONST_TRUE;
    // False || rhs -> rhs
    if (EQ(lhs, CONST_FALSE)) return rhs;
    // lhs || False -> lhs
    if (EQ(rhs, CONST_FALSE)) return lhs;
  }

  return node->shared_from_this();
}
ExprPtr BoolSimplifier::visit(Logical* node) {
  auto lhs = api::simplify(node->lhs);
  auto rhs = api::simplify(node->rhs);

  LogicalType op = node->operation_type;

  if (op == LogicalType::EQ) {
    if (EQ(lhs, rhs)) {
      return CONST_TRUE;
    }
  }

  auto diff = api::simplify(lhs - rhs);
  if (diff->get_type() == IRNodeType::Const) {
    switch (op) {
#define TYPE_LOGICALTYPE_OP_MAP(opname, op) \
  case LogicalType::opname:                 \
    if (diff op 0) {                        \
      return CONST_TRUE;                    \
    } else {                                \
      return CONST_FALSE;                   \
    }
#include "x/logical_types.def"
      default:
        break;
    }
  }
  if (diff->get_type() == IRNodeType::Unary) {
    CAST_TO(diffu, Unary, diff);
    if (diffu->operand->get_type() == IRNodeType::Const) {
      switch (op) {
#define TYPE_LOGICALTYPE_OP_MAP(opname, op) \
  case LogicalType::opname:                 \
    if (diff op 0) {                        \
      return CONST_TRUE;                    \
    } else {                                \
      return CONST_FALSE;                   \
    }
#include "x/logical_types.def"
        default:
          break;
      }
    }
  }
  return std::make_shared<Logical>(lhs, rhs, node->operation_type);
}  // namespace simplify

ExprPtr BoolSimplifier::visit(Cast* node) {
  return node->shared_from_this();
  // To be implemented
}

}  // namespace simplify
}  // namespace ir
