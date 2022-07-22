#include "IR/Expr.h"
#include "Pass/Common/Simplify.h"
#include "api.h"

namespace ir {
namespace simplify {

#define DECLR_BIN_LRHS(node, nlhs, nrhs) \
  auto nlhs = node->lhs;                 \
  auto nrhs = node->rhs;

void PatternSimplifier::setRules(IRNodeType ntype,
                                 std::list<SimplRule>& rules) {
  pattern_rules_[ntype] = rules;
}

void PatternSimplifier::addRule(IRNodeType ntype, SimplRule& rule) {
  pattern_rules_[ntype].push_back(rule);
}

NodePtr PatternSimplifier::simplifyNode(Node* node) {
  PatternSimplifier simplifier;
  return simplifier.visit(node);
}

#define VISIT_CAST(n) (ir::ptr_cast<Expr>(visit(n.get())))
#define RET(expr) return expr;
#define RETR(expr) return ir::ptr_cast<Expr>(simpl->visit((expr).get()));

// ExprPtr PatternSimplifier::visit(ScalarVar* node) {
//   ExprPtr expr = node->shared_from_this();

//   for (SimplRule& rule : pattern_rules_[IRNodeType::ScalarVar])
//     if (ExprPtr res = rule(this, expr)) return res;
//   return expr;
// }

ExprPtr PatternSimplifier::visit(Unary* node) {
  ExprPtr opd = VISIT_CAST(node->operand);
  ExprPtr expr = std::make_shared<Unary>(opd, node->operation_type);

  for (SimplRule& rule : pattern_rules_[IRNodeType::Unary])
    if (ExprPtr res = rule(this, expr)) return res;
  return expr;
}

ExprPtr PatternSimplifier::visit(Binary* node) {
  ExprPtr lhs = VISIT_CAST(node->lhs);
  ExprPtr rhs = VISIT_CAST(node->rhs);
  ExprPtr expr = std::make_shared<Binary>(lhs, rhs, node->operation_type);

  for (SimplRule& rule : pattern_rules_[IRNodeType::Binary])
    if (ExprPtr res = rule(this, expr)) return res;
  return expr;
}

ExprPtr PatternSimplifier::visit(Cast* node) {
  ExprPtr opd = VISIT_CAST(node->expr_);
  ExprPtr expr = std::make_shared<Cast>(opd, node->get_dtype());

  for (SimplRule& rule : pattern_rules_[IRNodeType::Cast])
    if (ExprPtr res = rule(this, expr)) return res;
  return expr;
}

#define OP(node, op) (node->operation_type == BinaryType::op)

#define SIMPL_RULE(NODE, RULE)                            \
  [](PatternSimplifier* simpl, ExprPtr expr) -> ExprPtr { \
    if (expr->get_type() == IRNodeType::NODE) {           \
      CAST_TO(node, NODE, expr);                          \
      RULE                                                \
    }                                                     \
    return nullptr;                                       \
  }

#define VAR_PR(RULE)                \
  SIMPL_RULE(ScalarVar, {           \
    auto dtype = node->get_dtype(); \
    RULE                            \
  })

#define UNA_PR(RULE)                \
  SIMPL_RULE(Unary, {               \
    auto dtype = node->get_dtype(); \
    auto op = node->operation_type; \
    auto opd = node->operand;       \
    RULE                            \
  })

#define BIN_PR(RULE)                      \
  SIMPL_RULE(Binary, {                    \
    DECLR_BIN_LRHS(node, lhs, rhs);       \
    BinaryType op = node->operation_type; \
    ScalarType dtype = lhs->get_dtype();  \
    RULE                                  \
  })

#define BIN_LB_PR(RULE)               \
  BIN_PR(if (NTYPE(lhs, Binary)) {    \
    CAST_TO(lbin, Binary, lhs);       \
    DECLR_BIN_LRHS(lbin, llhs, lrhs); \
    RULE                              \
  })

#define BIN_RB_PR(RULE)               \
  BIN_PR(if (NTYPE(rhs, Binary)) {    \
    CAST_TO(rbin, Binary, rhs);       \
    DECLR_BIN_LRHS(rbin, rlhs, rrhs); \
    RULE                              \
  })

#define BIN_B_B_PR(RULE)                                 \
  BIN_PR(if (NTYPE(lhs, Binary) && NTYPE(rhs, Binary)) { \
    CAST_TO(lbin, Binary, lhs);                          \
    CAST_TO(rbin, Binary, rhs);                          \
    DECLR_BIN_LRHS(lbin, llhs, lrhs);                    \
    DECLR_BIN_LRHS(rbin, rlhs, rrhs);                    \
    RULE                                                 \
  })

#define IMPLEMENT_PATTERN_SIMPLIFIER(SNAME) \
  NodePtr SNAME::simplifyNode(Node* node) { \
    SNAME simplifier;                       \
    return simplifier.visit(node);          \
  }                                         \
                                            \
  SNAME::SNAME()

// IMPLEMENT_PATTERN_SIMPLIFIER(CoefficientGetter) {
//   std::list<SimplRule> cg_var_prules = {VAR_PR(RET(ONE))};
//   setRules(IRNodeType::ScalarVar, cg_var_prules);
// }

IMPLEMENT_PATTERN_SIMPLIFIER(SingleVarSimplifier) {
  std::list<SimplRule> sv_una_prules = {
    UNA_PR(  // -0 = 0
        if ((op == UnaryType::Negate) && EQ(opd, ZERO)) RET(ZERO)),
    UNA_PR(  // -(c * x) = (-c) * x
        if (op == UnaryType::Negate && NTYPE(opd, Binary)) {
          CAST_TO(bin, Binary, opd);
          DECLR_BIN_LRHS(bin, lhs, rhs);
          if (bin->operation_type == BinaryType::Mul && NTYPE(lhs, Const))
            RET(PatternSimplifier::eval(lhs, UnaryType::Negate) * rhs)
        }),
    UNA_PR(  // -(-x) = x
        if (op == UnaryType::Negate && NTYPE(opd, Unary)) {
          CAST_TO(una, Unary, opd);
          auto uopd = una->operand;
          auto uop = una->operation_type;
          if (uop == UnaryType::Negate) RET(uopd)
        }),
  };

  setRules(IRNodeType::Unary, sv_una_prules);

  std::list<SimplRule> sv_bin_prules = {
    BIN_PR(  // (-x) - y = - (x + y)
        if (NTYPE(lhs, Unary)) {
          CAST_TO(luna, Unary, lhs);
          auto lopd = luna->operand;
          auto lop = luna->operation_type;
          if (op == BinaryType::Sub && lop == UnaryType::Negate)
            RET(-(rhs + lopd))
        }),

    BIN_PR(  // (-x) + x = 0
        if (NTYPE(lhs, Unary)) {
          CAST_TO(luna, Unary, lhs);
          auto lopd = luna->operand;
          auto lop = luna->operation_type;
          if (op == BinaryType::Add && lop == UnaryType::Negate &&
              EQ(rhs, lopd))
            RET(ZERO)
        }),

    //// Binary (Expr, Expr) ////

    BIN_PR(  // const ~ const
        if (NTYPE(lhs, Const) && NTYPE(rhs, Const))
            RET(PatternSimplifier::eval(lhs, rhs, op))),
    BIN_PR(  // a + 0 = a
        if (OP(node, Add) && EQ(rhs, ZERO)) RET(lhs)),
    BIN_PR(  // 0 + a = a
        if (OP(node, Add) && EQ(lhs, ZERO)) RET(rhs)),
    BIN_PR(  // a + a = a * 2
        if (OP(node, Add) && EQ(lhs, rhs)) RET(lhs * TWO)),
    BIN_PR(  // a - a = 0
        if (OP(node, Sub) && EQ(lhs, rhs)) RET(ZERO)),
    BIN_PR(  // a - 0 = a
        if (OP(node, Sub) && EQ(rhs, ZERO)) RET(lhs)),
    BIN_PR(  // a * 1 = a
        if (OP(node, Mul) && EQ(rhs, ONE)) RET(lhs)),
    BIN_PR(  // a / 1 = a
        if (OP(node, Div) && EQ(rhs, ONE)) RET(lhs)),
    BIN_PR(  // 1 * a = a
        if (OP(node, Mul) && EQ(lhs, ONE)) RET(rhs)),
    BIN_PR(  // a / a = 1
        if (OP(node, Div) && EQ(lhs, rhs)) RET(ONE)),
    BIN_PR(  // 0 / a = 0
        if (OP(node, Div) && EQ(lhs, ZERO)) RET(ZERO)),
    BIN_PR(  // a * c = c * a
        if (NTYPE(rhs, Const)) RET(rhs * lhs)),
    BIN_PR(  // a % 1 = 0
        if (OP(node, Mod) && EQ(rhs, ONE)) RET(ZERO)),

#define OPS(nop, lop, rop) (OP(node, nop) && OP(lbin, lop) && OP(rbin, rop))

    //// Binary (Binary, Binary) ////

    BIN_B_B_PR(  // (x - y) + (y - z) = x - z
        if (OPS(Add, Sub, Sub) && EQ(lrhs, rlhs)) RETR(llhs - rrhs)),
    BIN_B_B_PR(  // (x - y) + (z - x) = z - y
        if (OPS(Add, Sub, Sub) && EQ(llhs, rrhs)) RETR(rlhs - lrhs)),
    BIN_B_B_PR(  // x * y + x * z = x * (y + z)
        if (OPS(Add, Mul, Mul) && EQ(llhs, rlhs)) RETR(llhs * (lrhs + rrhs))),
    BIN_B_B_PR(  // y * x + x * z = x * (y + z)
        if (OPS(Add, Mul, Mul) && EQ(lrhs, rlhs)) RETR(lrhs * (llhs + rlhs))),
    BIN_B_B_PR(  // x * y + z * x = x * (y + z)
        if (OPS(Add, Mul, Mul) && EQ(llhs, rrhs)) RETR(llhs * (lrhs + rlhs))),
    BIN_B_B_PR(  // y * x + z * x = x * (y + z)
        if (OPS(Add, Mul, Mul) && EQ(lrhs, rrhs)) RETR(lrhs * (llhs + rlhs))),

    BIN_B_B_PR(  // (x + y) - (x + z) = y - z
        if (OPS(Sub, Add, Add) && EQ(llhs, rlhs)) RETR(lrhs - rrhs)),
    BIN_B_B_PR(  // (x - y) - (x - z) = z - y
        if (OPS(Sub, Sub, Sub) && EQ(llhs, rlhs)) RETR(rrhs - lrhs)),
    BIN_B_B_PR(  // (x + y) - (z + x) = y - z
        if (OPS(Sub, Add, Add) && EQ(llhs, rrhs)) RETR(lrhs - rlhs)),
    BIN_B_B_PR(  // (y + x) - (z + x) = y - z
        if (OPS(Sub, Add, Add) && EQ(lrhs, rrhs)) RETR(llhs - rlhs)),
    BIN_B_B_PR(  // (y + x) - (x + z) = y - z
        if (OPS(Sub, Add, Add) && EQ(lrhs, rlhs)) RETR(llhs - rrhs)),
    BIN_B_B_PR(  // x * y - x * z = x * (y - z)
        if (OPS(Sub, Mul, Mul) && EQ(llhs, rlhs)) RETR(llhs * (lrhs - rrhs))),
    BIN_B_B_PR(  // y * x - x * z = x * (y - z)
        if (OPS(Sub, Mul, Mul) && EQ(lrhs, rlhs)) RETR(lrhs * (llhs - rrhs))),
    BIN_B_B_PR(  // x * y - z * x = x * (y - z)
        if (OPS(Sub, Mul, Mul) && EQ(llhs, rrhs)) RETR(llhs * (lrhs - rlhs))),
    BIN_B_B_PR(  // y * x - z * x = x * (y - z)
        if (OPS(Sub, Mul, Mul) && EQ(lrhs, rrhs)) RETR(lrhs * (llhs - rlhs))),

#undef OPS
#define OPS(nop, lop) (OP(node, nop) && OP(lbin, lop))

    //// Binary (Binary, Expr) ////

    BIN_LB_PR(  // (x - y) + y = x
        if (OPS(Add, Sub) && EQ(lrhs, rhs)) RET(llhs)),
    BIN_LB_PR(  // (x - y) + y = x
        if (OPS(Add, Sub) && EQ(lrhs, rhs)) RET(llhs)),
    BIN_LB_PR(  // x * y + x = x * (y + 1)
        if (OPS(Add, Mul) && EQ(llhs, rhs)) RETR(rhs * (lrhs + ONE))),
    BIN_LB_PR(  // y * x + x = x * (y + 1)
        if (OPS(Add, Mul) && EQ(lrhs, rhs)) RETR(rhs * (llhs + ONE))),
    BIN_LB_PR(  // (x + c1) + c2 = x + (c1 + c2)
        if (OPS(Add, Add) && NTYPE(lrhs, Const) && NTYPE(rhs, Const))
            RET(llhs + PatternSimplifier::eval(lrhs, rhs, BinaryType::Add))),

    BIN_LB_PR(  // (x + y) - y = x
        if (OPS(Sub, Add) && EQ(lrhs, rhs)) RET(llhs)),
    BIN_LB_PR(  // (x + y) - x = y
        if (OPS(Sub, Add) && EQ(llhs, rhs)) RET(lrhs)),
    BIN_LB_PR(  // (x - y) - x = -y
        if (OPS(Sub, Sub) && EQ(llhs, rhs)) RET(ZERO - lrhs)),
    BIN_LB_PR(  // x * y - x = x * (y - 1)
        if (OPS(Sub, Mul) && EQ(llhs, rhs)) RETR(rhs * (lrhs + ONE))),
    BIN_LB_PR(  // y * x - x = x * (y - 1)
        if (OPS(Sub, Mul) && EQ(lrhs, rhs)) RETR(rhs * (lrhs + ONE))),

    BIN_LB_PR(  // (x * y) / x = y
        if (OPS(Div, Mul) && EQ(llhs, rhs)) RET(lrhs)),
    BIN_LB_PR(  // (x * y) / y = x
        if (OPS(Div, Mul) && EQ(lrhs, rhs)) RET(llhs)),
    BIN_LB_PR(  // (x + c1) - c2 = x + (c1 - c2)
        if (OPS(Sub, Add) && NTYPE(lrhs, Const) && NTYPE(rhs, Const))
            RET(llhs + PatternSimplifier::eval(lrhs, rhs, BinaryType::Sub))),
    BIN_LB_PR(  // (c1 * x) * c2 = (c1 * c2) * x
        if (OPS(Mul, Mul) && NTYPE(llhs, Const) && NTYPE(rhs, Const))
            RET(PatternSimplifier::eval(llhs, rhs, BinaryType::Mul) * lrhs)),
    BIN_LB_PR(  // (a * x) * (b * x) / x = a * (b * x)
        if (OPS(Div, Mul) && NTYPE(llhs, Binary) && NTYPE(lrhs, Binary)) {
          CAST_TO(llbin, Binary, llhs);
          CAST_TO(lrbin, Binary, lrhs);
          DECLR_BIN_LRHS(llbin, lllhs, llrhs);
          DECLR_BIN_LRHS(lrbin, lrlhs, lrrhs);
          if (EQ(llrhs, lrrhs) && EQ(llrhs, rhs)) RETR((lllhs * lrlhs) * rhs)
        }),

#undef OPS
#define OPS(nop, rop) (OP(node, nop) && OP(rbin, rop))

    //// Binary (Expr, Binary) ////

    BIN_RB_PR(  // x + (y - x) = y
        if (OPS(Add, Sub) && EQ(lhs, rrhs)) RET(rlhs)),
    BIN_RB_PR(  // x + x * y = x * (y + 1)
        if (OPS(Add, Mul) && EQ(lhs, rlhs)) RETR(lhs * (rrhs + ONE))),
    BIN_RB_PR(  // x + y * x = x * (y + 1)
        if (OPS(Add, Mul) && EQ(lhs, rrhs)) RETR(lhs * (rlhs + ONE))),
    BIN_RB_PR(  // x - (y + x) = 0 - y
        if (OPS(Sub, Add) && EQ(lhs, rrhs)) RETR(ZERO - rlhs)),
    BIN_RB_PR(  // x - (x + y) = 0 - y
        if (OPS(Sub, Add) && EQ(lhs, rlhs)) RETR(ZERO - rrhs)),
    BIN_RB_PR(  // x - (x - y) = y
        if (OPS(Sub, Sub) && EQ(lhs, rlhs)) RETR(rrhs)),
    BIN_RB_PR(  // x - y * x, x * (1 - y)
        if (OPS(Sub, Mul) && EQ(lhs, rrhs)) RETR(lhs * (ONE - rlhs))),
    BIN_RB_PR(  // x - x * y, x * (1 - y)
        if (OPS(Sub, Mul) && EQ(lhs, rlhs)) RETR(lhs * (ONE - rlhs))),
    BIN_RB_PR(  // x * (y - y) = 0
        if (OPS(Mul, Sub) && EQ(rlhs, rrhs)) RETR(ZERO)),
    BIN_RB_PR(  // c1 * (c2 * x) = (c1 * c2) * x
        if (OPS(Mul, Mul) && NTYPE(lhs, Const) && NTYPE(rlhs, Const))
            RET(PatternSimplifier::eval(lhs, rlhs, BinaryType::Mul) * rrhs)),
    BIN_RB_PR(  // x * ((-y) * z) = -(x * y * z)
        if (OPS(Mul, Mul) && NTYPE(rlhs, Unary)) {
          CAST_TO(rluna, Unary, rlhs);
          if (rluna->operation_type == UnaryType::Negate)
            RETR(-(lhs * rluna->operand * rrhs))
        }),

#undef OPS
  };
  setRules(IRNodeType::Binary, sv_bin_prules);
}  // namespace simplify

IMPLEMENT_PATTERN_SIMPLIFIER(PreSimplifier) {}

IMPLEMENT_PATTERN_SIMPLIFIER(PostSimplifier) {
  std::list<SimplRule> sv_una_prules = {
    UNA_PR(  // -0 = 0
        if ((op == UnaryType::Negate) && EQ(opd, ZERO)) RET(ZERO)),
    UNA_PR(  // -(c * x) = (-c) * x
        if (op == UnaryType::Negate && NTYPE(opd, Binary)) {
          CAST_TO(bin, Binary, opd);
          DECLR_BIN_LRHS(bin, lhs, rhs);
          if (bin->operation_type == BinaryType::Mul && NTYPE(lhs, Const))
            RET(PatternSimplifier::eval(lhs, UnaryType::Negate) * rhs)
        }),
  };
  setRules(IRNodeType::Unary, sv_una_prules);

  std::list<SimplRule> ps_bin_rules = {
    BIN_PR(  // x + (-y) = x - y
        if (NTYPE(rhs, Unary)) {
          CAST_TO(runary, Unary, rhs);
          auto ropd = runary->operand;
          auto rop = runary->operation_type;
          if (op == BinaryType::Add && rop == UnaryType::Negate) RET(lhs - ropd)
        }),
    BIN_PR(  // (-x) + y = y - x
        if (NTYPE(lhs, Unary)) {
          CAST_TO(luna, Unary, lhs);
          auto lopd = luna->operand;
          auto lop = luna->operation_type;
          if (op == BinaryType::Add && lop == UnaryType::Negate) RET(rhs - lopd)
        }),
    BIN_PR(  // (-x) - y = - (x + y)
        if (NTYPE(lhs, Unary)) {
          CAST_TO(luna, Unary, lhs);
          auto lopd = luna->operand;
          auto lop = luna->operation_type;
          if (op == BinaryType::Sub && lop == UnaryType::Negate)
            RET(-(rhs + lopd))
        }),
    BIN_PR(  // (-y) * z = - (y * z)
        if (NTYPE(lhs, Unary)) {
          CAST_TO(luna, Unary, lhs);
          auto lopd = luna->operand;
          auto lop = luna->operation_type;
          if (op == BinaryType::Mul && lop == UnaryType::Negate)
            RETR(-(lopd * rhs))
        }),
    BIN_PR(  // y * (-z) = - (y * z)
        if (NTYPE(rhs, Unary)) {
          CAST_TO(runa, Unary, rhs);
          auto ropd = runa->operand;
          auto rop = runa->operation_type;
          if (op == BinaryType::Mul && rop == UnaryType::Negate)
            RETR(-(ropd * lhs))
        }),
    BIN_PR(  // const ~ const
        if (NTYPE(lhs, Const) && NTYPE(rhs, Const))
            RET(PatternSimplifier::eval(lhs, rhs, op))),
    BIN_PR(  // (a * x) / a = x
        if (NTYPE(lhs, Binary)) {
          CAST_TO(lbin, Binary, lhs);
          DECLR_BIN_LRHS(lbin, llhs, lrhs)
          if (OP(node, Div) && OP(lbin, Mul) && EQ(llhs, rhs)) RETR(lrhs)
        }),
    BIN_PR(  // a + 0 = a
        if (OP(node, Add) && EQ(rhs, ZERO)) RET(lhs)),
    BIN_PR(  // 0 + a = a
        if (OP(node, Add) && EQ(lhs, ZERO)) RET(rhs)),
    BIN_PR(  // a - 0 = a
        if (OP(node, Sub) && EQ(rhs, ZERO)) RET(lhs)),
    BIN_PR(  // 0 - a = -a
        if (OP(node, Sub) && EQ(lhs, ZERO)) RET(-rhs)),
    BIN_PR(  // a * 0 = 0
        if (OP(node, Mul) && EQ(rhs, ZERO)) RET(ZERO)),
    BIN_PR(  // 0 * a = 0
        if (OP(node, Mul) && EQ(lhs, ZERO)) RET(ZERO)),
    BIN_PR(  // a % 1 = 0
        if (OP(node, Mod) && EQ(rhs, ONE)) RET(ZERO)),
    BIN_PR(  // a - a = 0
        if (OP(node, Sub) && EQ(lhs, rhs)) RET(ZERO)),
    BIN_PR(  // a / 1 = a
        if (OP(node, Div) && EQ(rhs, ONE)) RET(lhs)),
    BIN_PR(  // 0 / a = 0
        if (OP(node, Div) && EQ(lhs, ZERO)) RET(ZERO)),
    BIN_PR(  // (c1 * i1 + c2 * i2) / c3
             // = (c1 / c3 * i1 + c2 / c3 * i2)
        if (NTYPE(lhs, Binary)) {
          CAST_TO(lbin, Binary, lhs);
          DECLR_BIN_LRHS(lbin, llhs, lrhs);
          if (NTYPE(llhs, Binary) && NTYPE(lrhs, Binary)) {
            CAST_TO(llbin, Binary, llhs);
            CAST_TO(lrbin, Binary, lrhs);
            DECLR_BIN_LRHS(llbin, lllhs, llrhs);
            DECLR_BIN_LRHS(lrbin, lrlhs, lrrhs);
            if (OP(node, Div) && OP(lbin, Add) && OP(llbin, Mul) &&
                OP(lrbin, Mul) && NTYPE(lllhs, Const) && NTYPE(lrlhs, Const) &&
                NTYPE(rhs, Const)) {
              return api::simplify((lllhs / rhs * llrhs + lrlhs / rhs * lrrhs));
            }
          }
        }),
    BIN_PR(                        // (((x * x) * y) * y) / (x * y)
        if (NTYPE(lhs, Binary)) {  // (((x * x) * y) * y)
          CAST_TO(lbin, Binary, lhs);
          DECLR_BIN_LRHS(lbin, llhs, lrhs);  // ((x * x) * y) y

          if (NTYPE(llhs, Binary)) {
            CAST_TO(llbin, Binary, llhs);
            DECLR_BIN_LRHS(llbin, lllhs, llrhs);  // x * x y

            if (NTYPE(lllhs, Binary)) {
              CAST_TO(lllbin, Binary, lllhs);
              DECLR_BIN_LRHS(lllbin, llllhs, lllrhs);  // x x

              if (NTYPE(rhs, Binary)) {
                CAST_TO(rbin, Binary, rhs);
                DECLR_BIN_LRHS(rbin, rlhs, rrhs);  // x y

                if (EQ(llllhs, rlhs) && EQ(llrhs, rrhs)) {
                  if (OP(node, Div) && OP(lbin, Mul) && OP(llbin, Mul) &&
                      OP(lllbin, Mul) && OP(rbin, Mul)) {
                    return (lllrhs * lrhs);
                  }
                }
              }
            }
          }
        }),
    BIN_PR(  // (((((x * x) * y) * y) * z) * z) / ((x * y) * z)
        if (NTYPE(lhs, Binary)) {  // (((((x * x) * y) * y) * z) * z)
          CAST_TO(lbin, Binary, lhs);
          DECLR_BIN_LRHS(lbin, llhs, lrhs);  // ((((x * x) * y) * y) * z) z

          if (NTYPE(llhs, Binary)) {
            CAST_TO(llbin, Binary, llhs);
            DECLR_BIN_LRHS(llbin, lllhs, llrhs);  // ((((x * x) * y) * y) z

            if (NTYPE(lllhs, Binary)) {
              CAST_TO(lllbin, Binary, lllhs);
              DECLR_BIN_LRHS(lllbin, llllhs, lllrhs);  // ((x * x) * y) y

              if (NTYPE(llllhs, Binary)) {
                CAST_TO(llllbin, Binary, llllhs);
                DECLR_BIN_LRHS(llllbin, lllllhs, llllrhs);  // x * x y

                if (NTYPE(lllllhs, Binary)) {
                  CAST_TO(lllllbin, Binary, lllllhs);
                  DECLR_BIN_LRHS(lllllbin, llllllhs, lllllrhs);  // x x

                  if (NTYPE(rhs, Binary)) {
                    CAST_TO(rbin, Binary, rhs);
                    DECLR_BIN_LRHS(rbin, rlhs, rrhs);  // x * y z

                    if (NTYPE(rlhs, Binary)) {
                      CAST_TO(rlbin, Binary, rlhs);
                      DECLR_BIN_LRHS(rlbin, rllhs, rlrhs);  // x y

                      if (EQ(llllllhs, rllhs) && EQ(llllrhs, rlrhs) &&
                          EQ(lrhs, rrhs)) {
                        if (OP(node, Div) && OP(lbin, Mul) && OP(llbin, Mul) &&
                            OP(lllbin, Mul) && OP(llllbin, Mul) &&
                            OP(lllllbin, Mul) & OP(rbin, Mul) &
                                OP(rlbin, Mul)) {
                          return (lllllrhs * lllrhs * llrhs);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }),
    BIN_PR(  // (x*x*y*y*z*z*q*q)/(x*y*z*q)
        if (NTYPE(lhs, Binary)) {
          CAST_TO(lbin, Binary, lhs);
          DECLR_BIN_LRHS(lbin, llhs, lrhs);

          if (NTYPE(llhs, Binary)) {
            CAST_TO(llbin, Binary, llhs);
            DECLR_BIN_LRHS(llbin, lllhs, llrhs);

            if (NTYPE(lllhs, Binary)) {
              CAST_TO(lllbin, Binary, lllhs);
              DECLR_BIN_LRHS(lllbin, llllhs, lllrhs);

              if (NTYPE(llllhs, Binary)) {
                CAST_TO(llllbin, Binary, llllhs);
                DECLR_BIN_LRHS(llllbin, lllllhs, llllrhs);

                if (NTYPE(lllllhs, Binary)) {
                  CAST_TO(lllllbin, Binary, lllllhs);
                  DECLR_BIN_LRHS(lllllbin, llllllhs, lllllrhs);

                  if (NTYPE(llllllhs, Binary)) {
                    CAST_TO(llllllbin, Binary, llllllhs);
                    DECLR_BIN_LRHS(llllllbin, lllllllhs, llllllrhs);

                    if (NTYPE(lllllllhs, Binary)) {
                      CAST_TO(lllllllbin, Binary, lllllllhs);
                      DECLR_BIN_LRHS(lllllllbin, llllllllhs, lllllllrhs);

                      if (NTYPE(rhs, Binary)) {
                        CAST_TO(rbin, Binary, rhs);
                        DECLR_BIN_LRHS(rbin, rlhs, rrhs);

                        if (NTYPE(rlhs, Binary)) {
                          CAST_TO(rlbin, Binary, rlhs);
                          DECLR_BIN_LRHS(rlbin, rllhs, rlrhs);

                          if (NTYPE(rllhs, Binary)) {
                            CAST_TO(rllbin, Binary, rllhs);
                            DECLR_BIN_LRHS(rllbin, rlllhs, rllrhs);

                            if (EQ(llllllllhs, rlllhs) &&
                                EQ(llllllrhs, rllrhs) && EQ(llllrhs, rlrhs) &&
                                EQ(lrhs, rrhs)) {
                              if (OP(node, Div) && OP(lbin, Mul) &&
                                  OP(llbin, Mul) && OP(lllbin, Mul) &&
                                  OP(llllbin, Mul) && OP(lllllbin, Mul) &&
                                  OP(llllllbin, Mul) && OP(lllllllbin, Mul) &&
                                  OP(rbin, Mul) && OP(rlbin, Mul) &&
                                  OP(rllbin, Mul)) {
                                return (lllllllrhs * lllllrhs * lllrhs * llrhs);
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }),
    BIN_PR(                       // -((-1 * a) * a) / a = a
        if (NTYPE(lhs, Unary)) {  // -((-1 * a) * a)
          auto op = ir::ptr_cast<Unary>(lhs)->operation_type;
          auto opd = ir::ptr_cast<Unary>(lhs)->operand;
          if (NTYPE(opd, Binary)) {  // (-1 * a) * a
            CAST_TO(obin, Binary, opd);
            DECLR_BIN_LRHS(obin, olhs, orhs);  // (-1 * a) a
            if (OP(node, Div) && OP(obin, Mul) && EQ(orhs, rhs)) {
              return (std::make_shared<Unary>(olhs, op));
            }
          }
        }),
    BIN_PR(  // -(a*b*c + 2*a*b*c) / (a*b*c)
        if (NTYPE(lhs, Unary)) {
          auto op = ir::ptr_cast<Unary>(lhs)->operation_type;
          auto opd = ir::ptr_cast<Unary>(lhs)->operand;
          if (NTYPE(opd, Binary)) {  // a*b*c * 2*a*b*c
            CAST_TO(obin, Binary, opd);
            DECLR_BIN_LRHS(obin, olhs, orhs);  // a*b*c 2*a*b*c

            if (OP(node, Div) && OP(obin, Mul) && op == UnaryType::Negate &&
                EQ(olhs, rhs)) {
              return std::make_shared<Unary>(orhs, op);
            }
          }
        }),
    BIN_PR(  // -(-(a*b*c + 2*a*b*c)) / (a*b*c)
        if (NTYPE(lhs, Unary)) {
          auto op = ir::ptr_cast<Unary>(lhs)->operation_type;
          auto opd = ir::ptr_cast<Unary>(lhs)->operand;
          if (NTYPE(opd, Unary)) {
            auto oop = ir::ptr_cast<Unary>(opd)->operation_type;
            auto oopd = ir::ptr_cast<Unary>(opd)->operand;
            if (NTYPE(oopd, Binary)) {  // a*b*c * 2*a*b*c
              CAST_TO(obin, Binary, oopd);
              DECLR_BIN_LRHS(obin, olhs, orhs);  // a*b*c 2*a*b*c

              if (OP(node, Div) && OP(obin, Mul) && op == UnaryType::Negate &&
                  oop == UnaryType::Negate && EQ(olhs, rhs)) {
                return orhs;
              }
            }
          }
        }),
    BIN_PR(  // (0 - (-1 * a)) / b = a / b
        if (NTYPE(lhs, Binary)) {
          CAST_TO(lbin, Binary, lhs);
          DECLR_BIN_LRHS(lbin, llhs, lrhs);  // 0 (-1 * a)
          if (NTYPE(lrhs, Binary)) {
            CAST_TO(lrbin, Binary, lrhs);
            DECLR_BIN_LRHS(lrbin, lrlhs, lrrhs);  // -1 a
            if (NTYPE(lrlhs, Unary)) {
              auto op = ir::ptr_cast<Unary>(lrlhs)->operation_type;
              auto opd = ir::ptr_cast<Unary>(lrlhs)->operand;
              if (OP(node, Div) && OP(lbin, Sub) && OP(lrbin, Mul) &&
                  EQ(llhs, ZERO) && op == UnaryType::Negate && EQ(opd, ONE)) {
                return api::simplify(lrrhs / rhs);
              }
            }
          }
        }),
#define OPS(nop, rop) (OP(node, nop) && OP(rbin, rop))

    //// Binary (Expr, Binary) ////
    BIN_RB_PR(  // x * (y - y) = 0
        if (OPS(Mul, Sub) && EQ(rlhs, rrhs)) RETR(ZERO)),

#undef OPS
  };
  setRules(IRNodeType::Binary, ps_bin_rules);
}  // namespace simplify

// Some old simplify pattern rules
SimplRule custom_pr1 = BIN_PR(  // (x + (y + z)) - y = (x + z)
    if (NTYPE(lhs, Binary)) {
      CAST_TO(lbin, Binary, lhs);
      DECLR_BIN_LRHS(lbin, llhs, lrhs);

      if (NTYPE(lrhs, Binary)) {
        CAST_TO(lrbin, Binary, lrhs);
        DECLR_BIN_LRHS(lrbin, lrlhs, lrrhs);

        if (OP(node, Sub) && OP(lbin, Add) && OP(lrbin, Add) && EQ(lrlhs, rhs))
          RETR(llhs + lrrhs);
      }
    });

SimplRule custom_pr2 =
    BIN_PR(  // (((x + (y + z)) + u) - (x + y)) + w = (z + u + w)
        if (NTYPE(lhs, Binary)) {
          CAST_TO(lbin, Binary, lhs);
          DECLR_BIN_LRHS(lbin, llhs, lrhs);  // ((x + (y + z)) + u)  (x + y)

          if (NTYPE(llhs, Binary)) {
            CAST_TO(llbin, Binary, llhs);
            DECLR_BIN_LRHS(llbin, lllhs, llrhs);  // (x + (y + z)) u

            if (NTYPE(lllhs, Binary)) {
              CAST_TO(lllbin, Binary, lllhs);
              DECLR_BIN_LRHS(lllbin, llllhs, lllrhs);  // x (y + z)

              if (NTYPE(lllrhs, Binary)) {
                CAST_TO(lllrbin, Binary, lllrhs);
                DECLR_BIN_LRHS(lllrbin, lllrlhs, lllrrhs);  // y  z

                if (NTYPE(lrhs, Binary)) {
                  CAST_TO(lrbin, Binary, lrhs);
                  DECLR_BIN_LRHS(lrbin, lrlhs, lrrhs);  // x  y

                  if (OP(node, Add) && OP(lbin, Sub) && OP(llbin, Add) &&
                      OP(lrbin, Add) && OP(lllbin, Add) && OP(lllrbin, Add) &&
                      EQ(llllhs, lrlhs) && EQ(lllrlhs, lrrhs))
                    RETR(lllrrhs + llrhs + rhs);
                }
              }
            }
          }
        });

SimplRule custom_pr3 = BIN_PR(  // (u + (x + (y + z))) - (x + y) = u + z
    if (NTYPE(lhs, Binary)) {
      CAST_TO(lbin, Binary, lhs);
      DECLR_BIN_LRHS(lbin, llhs, lrhs);  // u (x + (y + z))

      if (NTYPE(lrhs, Binary)) {
        CAST_TO(lrbin, Binary, lrhs);
        DECLR_BIN_LRHS(lrbin, lrlhs, lrrhs);  // x (y + z)

        if (NTYPE(lrrhs, Binary)) {
          CAST_TO(lrrbin, Binary, lrrhs);
          DECLR_BIN_LRHS(lrrbin, lrrlhs, lrrrhs);  // y z

          if (NTYPE(rhs, Binary)) {
            CAST_TO(rbin, Binary, rhs);
            DECLR_BIN_LRHS(rbin, rlhs, rrhs);  // x  y

            if (OP(node, Sub) && OP(lbin, Add) && OP(lrbin, Add) &&
                OP(lrrbin, Add) && OP(rbin, Add) && EQ(lrlhs, rlhs) &&
                EQ(lrrlhs, rrhs))
              RETR(llhs + lrrrhs);
          }
        }
      }
    });

SimplRule custom_pr4 = BIN_PR(  // c * (x + y) = x * c + y * c
    if (NTYPE(rhs, Binary)) {
      CAST_TO(rbin, Binary, rhs);
      DECLR_BIN_LRHS(rbin, rlhs, rrhs);

      if (NTYPE(lhs, Const) && NTYPE(rlhs, IterVar) && NTYPE(rrhs, IterVar)) {
        if (OP(node, Mul) && OP(rbin, Add)) {
          return (rlhs * lhs + rrhs * lhs);
        }
      }
    });

#undef BIN_B_B_PR
#undef BIN_RB_PR
#undef BIN_LB_PR
#undef BIN_PR
#undef UNA_PR
#undef SIMPL_RULE

}  // namespace simplify
}  // namespace ir
