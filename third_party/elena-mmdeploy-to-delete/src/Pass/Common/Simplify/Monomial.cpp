#include "Pass/Common/Simplify.h"

namespace {

using namespace ir;            // NOLINT
using namespace ir::simplify;  // NOLINT

class ExprGenVisitor : public MutatorBase<ExprGenVisitor> {
 public:
  ExprGenVisitor(const std::map<ExprPtr, unsigned int,
                                simplify::Monomial::ExprStrLess>& var_map) {
    for (auto& p : var_map) {
      if (p.second == 0) continue;

      if (!expr_)
        expr_ = p.first;
      else
        expr_ = expr_ * p.first;

      for (unsigned int i = 1; i < p.second; i++) {
        expr_ = expr_ * p.first;
      }
    }
  }

  ExprPtr gen(ExprPtr node) { return ir::ptr_cast<Expr>(visit(node.get())); }

  ExprPtr visit(ScalarVar* node) {
    ELENA_ASSERT_EQ(node->get_name(),
                    "$elena::simplify::__internal::placeholder",
                    "The Var should be VarPH only");
    return expr_;
  }

  ExprPtr visit(Unary* node) {
    return std::make_shared<Unary>(
        ir::ptr_cast<Expr>(visit(node->operand.get())), node->operation_type);
  }

  ExprPtr visit(Binary* node) {
    return std::make_shared<Binary>(ir::ptr_cast<Expr>(visit(node->lhs.get())),
                                    ir::ptr_cast<Expr>(visit(node->rhs.get())),
                                    node->operation_type);
  }

  using MutatorBase::visit;

  ExprPtr expr_;
};

template <typename Map>
bool mapCompare(Map const& lhs, Map const& rhs) {
  // No predicate needed because there is operator== for pairs already.
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

}  // namespace

namespace ir {
namespace simplify {

#define NEW_VARPH(DTYPE)                                                   \
  std::make_shared<ScalarVar>("$elena::simplify::__internal::placeholder", \
                              DTYPE)

Monomial::Monomial(ExprPtr node) {
  if (node->get_type() == IRNodeType::Const) {
    expr_ = node;
  } else {
    var_map_[node] = 1;
    expr_ = NEW_VARPH(node->get_dtype());
  }
}

Monomial Monomial::operator+() const { return Monomial(*this); }

Monomial Monomial::operator-() const {
  Monomial mon;
  mon.var_map_ = var_map_;
  mon.expr_ = -expr_;
  mon.simpl();
  return mon;
}

Polynomial Monomial::operator+(const Monomial& b) const {
  if (sameVar(b)) {
    Monomial res(*this);
    res.expr_ = res.expr_ + b.expr_;
    res.simpl();
    return Polynomial(res);
  } else {
    Polynomial res(*this);
    res.exprs_.push_back(b);
    return res;
  }
}

Polynomial Monomial::operator-(const Monomial& b) const {
  if (sameVar(b)) {
    Monomial res(*this);
    res.expr_ = res.expr_ - b.expr_;
    res.simpl();
    return Polynomial(res);
  } else {
    Polynomial res(*this);
    auto nb = -b;
    res.exprs_.push_back(nb);
    return res;
  }
}

// Monomial Monomial::operator*(const Monomial& b) const {
//   Monomial res(*this);

//   // (a * _) * (b * _) = (a * b * _)
//   if (!isConstant() && !b.isConstant()) res.getCoefficient();

//   res.expr_ = res.expr_ * b.expr_;
//   for (const auto& p : b.var_map_)
//     res.var_map_[p.first] = res.var_map_[p.first] + p.second;

//   res.simpl();

//   return res;
// }

Monomial Monomial::operator*(const Monomial& b) const {
  Monomial res(*this);

  for (const auto& p : b.var_map_)
    res.var_map_[p.first] = res.var_map_[p.first] + p.second;
  res.expr_ = res.expr_ * b.expr_;

  // (a * _) * (b * _) = (a * b * _)
  if (!isConstant() && !b.isConstant())
    res.expr_ = res.expr_ / NEW_VARPH(expr_->get_dtype());

  res.simpl();
  return res;
}

bool Monomial::sameVar(const Monomial& mon) const {
  if (var_map_.size() != mon.var_map_.size()) return false;
  for (auto& a_kv : var_map_) {
    bool sv = false;
    for (auto& b_kv : mon.var_map_) {
      if (EQ(a_kv.first, b_kv.first) && a_kv.second == b_kv.second) {
        sv = true;
        break;
      }
    }
    if (!sv) return false;
  }
  return true;
  // We cannot use `map_compare` for the IterVar should be compared by EQ
  // instead of the address. Too bad!
  // return map_compare(var_map_, mon.var_map_);
}

bool Monomial::isConstant() const { return var_map_.empty(); }

Polynomial Monomial::asPoly() const { return Polynomial(*this); }

ExprPtr Monomial::asExpr() const {
  ExprGenVisitor expr_gen(var_map_);
  return expr_gen.gen(expr_);
}

ScalarType Monomial::getDtype() const { return expr_->get_dtype(); }

void Monomial::simpl() {
  expr_ = ::ir::ptr_cast<Expr>(SingleVarSimplifier::simplifyNode(expr_.get()));
  if (expr_->get_type() == IRNodeType::Const) var_map_.clear();
}

// void Monomial::getCoefficient() {
//   expr_ = ::ir::ptr_cast<Expr>(CoefficientGetter::simplifyNode(expr_.get()));
// }

int Monomial::isNormalForm() const {
  auto dtype = expr_->get_dtype();
  auto phvar = NEW_VARPH(dtype);

  if (expr_->get_type() == IRNodeType::Const) return 1;

  if (EQ(expr_, phvar)) return 2;

  if (expr_->get_type() == IRNodeType::Binary) {
    CAST_TO(bin, Binary, expr_);
    if (bin->lhs->get_type() == IRNodeType::Const && EQ(bin->rhs, phvar))
      return 3;
  }

  return 0;
}

}  // namespace simplify
}  // namespace ir
