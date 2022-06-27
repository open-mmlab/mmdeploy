#include "Pass/Common/Simplify.h"

namespace ir {
namespace simplify {

Polynomial::Polynomial(const Monomial& m) { exprs_.push_back(m); }

Polynomial Polynomial::operator+() const { return Polynomial(*this); }

Polynomial Polynomial::operator-() const {
  Polynomial poly;
  for (auto& mon : exprs_) {
    poly.exprs_.push_back(-mon);
  }
  return poly;
}

Polynomial Polynomial::operator+(const Polynomial& b) const {
  Polynomial poly(*this);
  for (const auto& mon : b.exprs_) {
    bool merged = false;
    for (size_t i = 0; i < poly.exprs_.size(); ++i) {
      if (poly.exprs_[i].sameVar(mon)) {
        poly.exprs_[i] = (poly.exprs_[i] + mon).asMonomial();
        merged = true;
        break;
      }
    }
    // simplify the polynomial every time, for zero constant monomial.
    if (merged)
      poly.simpl();
    else
      poly.exprs_.push_back(mon);
  }
  poly.simpl();
  return poly;
}

Polynomial Polynomial::operator-(const Polynomial& b) const {
  Polynomial poly(*this);
  for (const Monomial& mon : b.exprs_) {
    bool merged = false;
    for (size_t i = 0; i < poly.exprs_.size(); ++i) {
      if (poly.exprs_[i].sameVar(mon)) {
        poly.exprs_[i] = (poly.exprs_[i] - mon).asMonomial();
        merged = true;
        break;
      }
    }
    if (merged)
      poly.simpl();
    else
      poly.exprs_.push_back(-mon);
  }
  poly.simpl();
  return poly;
}

Polynomial Polynomial::operator*(const Monomial& b) const {
  Polynomial poly;

  if (EQ(b.asExpr(), PatternSimplifier::numWithType(0, b.getDtype()))) {
    poly.exprs_.push_back(b);
    return poly;
  }

  for (const auto& e : exprs_) poly.exprs_.push_back(e * b);
  return poly;
}

Polynomial Polynomial::operator*(const Polynomial& b) const {
  // if (!isMonomial() && !b.isMonomial())
  //   return Polynomial(Monomial(asExpr() * b.asExpr()));

  Polynomial poly;
  for (const auto& m : b.exprs_) poly = poly + (*this) * m;
  return poly;
}

Monomial Polynomial::asMonomial() const {
  ELENA_ASSERT(isMonomial(), "It is not a monomial.");
  return exprs_[0];
}

ExprPtr Polynomial::asExpr() const {
  ScalarType dtype = exprs_[0].getDtype();
  ExprPtr res = nullptr;

  for (unsigned int i = 0; i < exprs_.size(); ++i) {
    auto e = exprs_[i].asExpr();
    if (!EQ(e, PatternSimplifier::numWithType(0, dtype)))
      res = res ? res + e : e;
  }

  return res ? res : PatternSimplifier::numWithType(0, dtype);
}

bool Polynomial::isMonomial() const { return exprs_.size() == 1; }

void Polynomial::simpl() {
  if (exprs_.size() == 0) return;
  ScalarType dtype = exprs_[0].getDtype();

  std::vector<Monomial> exprs;
  exprs.push_back(Monomial(PatternSimplifier::numWithType(0, dtype)));
  for (const auto& mon : exprs_) {
    if (mon.var_map_.empty())
      exprs[0] = (exprs[0] + mon).asMonomial();
    else
      exprs.push_back(mon);
  }

  exprs_ = exprs;
}

}  // namespace simplify
}  // namespace ir
