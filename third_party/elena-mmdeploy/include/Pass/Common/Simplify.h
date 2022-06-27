#ifndef ELENA_INCLUDE_PASS_COMMON_SIMPLIFY_H_
#define ELENA_INCLUDE_PASS_COMMON_SIMPLIFY_H_

#include <cstdint>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "IR/Expr.h"
#include "IR/ExprEqual.h"
#include "IR/MutatorBase.h"
#include "IR/Type.h"
#include "api.h"

namespace ir {
namespace simplify {

class Polynomial;

/**
 * @brief The simplifier to simplify all the expressions in an AST
 *
 * This simplifier will find all the expressions in an abstract syntax tree and
 * simplify them. This simplifier will copy an AST instead of modify it.
 */
class ExprSimplifier : public MutatorBase<ExprSimplifier> {
 public:
  /**
   * @brief The interface to simplify an AST
   *
   * This static member method will find all the expressions in `node` and
   * simplify them by calling `SimplifyExpr`.
   */
  static NodePtr simplifyNode(Node* node);
  static NodePtr simplifyNode(NodePtr node);

  /**
   * @brief The interface of simplifying an expression
   *
   * This static member method will simplify the expression in three steps:
   * `PreSimplifier`, `BoolSimplifier` or `NumericSimplifier`, `PostSimplifier`.
   * The first and the last simplifiers are based on pattern rules. And
   * `NumericSimplifier` is based on monomial and polynomial. Read the
   * documentations of Simplify Pass for more details.
   */
  static ExprPtr simplifyExpr(Expr* expr);
  static ExprPtr simplifyExpr(ExprPtr expr);

  template <typename T>
  NodePtr visit(Array<T>* node);

  NodePtr visit(For* node);
  NodePtr visit(Block* node);
  NodePtr visit(Provide* node);
  NodePtr visit(Store* node);
  NodePtr visit(Range* node);
  NodePtr visit(Allocate* node);
  NodePtr visit(IfThenElse* node);
  NodePtr visit(Let* node);
  NodePtr visit(Attr* node);
  using MutatorBase::visit;
};

/**
 * @brief The simplifier for expressions with type `bool`
 */
class BoolSimplifier : public MutatorBase<BoolSimplifier> {
 public:
  /**
   * @brief The interface to simplify an AST
   *
   * This method will not check the data type of `node`. So make sure that it is
   * a boolean expression before invoking this.
   */
  static ExprPtr simplifyNode(Expr* node);

  ExprPtr visit(Unary* node);
  ExprPtr visit(Binary* node);
  ExprPtr visit(Logical* node);
  ExprPtr visit(Cast* node);
  using MutatorBase::visit;
};

/**
 * @brief The simplifier for numeric expressions
 *
 * This simplifier is based on monomial and polynomial.
 */
class NumericSimplifier : public VisitorBase<NumericSimplifier> {
 public:
  /**
   * @brief The interface to simplify an AST
   *
   * This method will not check the data type of `node`. So make sure that it is
   * a numeric expression before invoking this.
   */
  static ExprPtr simplifyExpr(Expr* node);

  template <typename T>
  void visit(Const<T>* node);

  void visit(IterVar* node);
  void visit(ScalarVar* node);
  void visit(TensorVar* node);

  void visit(Unary* node);
  void visit(Binary* node);
  void visit(Select* node);
  void visit(Cast* node);
  void visit(Call* node);
  using VisitorBase::visit;

 private:
  void push(Polynomial poly);
  Polynomial pop();

  std::stack<Polynomial> results_;
};

class PatternSimplifier;
using SimplRule = std::function<ExprPtr(PatternSimplifier*, ExprPtr)>;

/**
 * @brief The simplified base on pattern rules.
 *
 * An implementation of `PatternSimplifier` will add pattern rules to
 * `pattern_rules_`, which is a map of node type and list of `SimplRule`.
 * A `SimplRule` is a lambda expression with type `ExprPtr(PatternSimplifier*,
 * ExprPtr)`. Read the documentation of Simplify Pass for more details.
 */
class PatternSimplifier : public MutatorBase<PatternSimplifier> {
 public:
  PatternSimplifier() = default;

  static NodePtr simplifyNode(Node* node);
  static ExprPtr numWithType(int val, ScalarType dtype);
  static ExprPtr eval(ExprPtr opd, UnaryType op);
  static ExprPtr eval(ExprPtr lhs, ExprPtr rhs, BinaryType op);
  static ExprPtr eval(ExprPtr expr, ScalarType dst_type);

  void setRules(IRNodeType ntype, std::list<SimplRule>& rules);
  void addRule(IRNodeType ntype, SimplRule& rule);

  // ExprPtr visit(ScalarVar* node);
  ExprPtr visit(Unary* node);
  ExprPtr visit(Binary* node);
  ExprPtr visit(Cast* node);
  using MutatorBase::visit;

 protected:
  std::map<IRNodeType, std::list<SimplRule>> pattern_rules_;
};

/**
 * @brief The monomial expression
 *
 * The internal detail of `NumericSimplifier`, which should never be constructed
 * by users.
 */
class Monomial {
 public:
  friend class Polynomial;

  Monomial() = default;
  Monomial(const Monomial& mon) = default;

  template <typename T>
  explicit Monomial(ConstPtr<T> node) {
    expr_ = node;
  }

  explicit Monomial(ExprPtr node);

  Monomial operator+() const;
  Monomial operator-() const;

  Polynomial operator+(const Monomial& b) const;
  Polynomial operator-(const Monomial& b) const;
  Monomial operator*(const Monomial& b) const;

  bool sameVar(const Monomial& mon) const;

  bool isConstant() const;
  Polynomial asPoly() const;
  ExprPtr asExpr() const;

  ScalarType getDtype() const;

  struct ExprStrLess {
    bool operator()(const ExprPtr& lhs, const ExprPtr& rhs) const {
      std::ostringstream lhs_str;
      std::ostringstream rhs_str;
      api::dump_expr(lhs, lhs_str);
      api::dump_expr(rhs, rhs_str);
      // std::cout << "    " << lhs_str.str() << ", " << rhs_str.str()
      //           << std::endl;
      return lhs_str.str() < rhs_str.str();
    }
  };

  // Set the VARPH to 1 in `expr_`
  void getCoefficient();

 private:
  // Simplify the monomial using ExprSimplifier
  void simpl();
  // 1: c; 2: _; 3: c * _; 0: otherwise
  int isNormalForm() const;

  std::map<ExprPtr, unsigned int, ExprStrLess> var_map_;
  ExprPtr expr_ = nullptr;
};

/**
 * @brief The polynomial expression
 *
 * The internal detail of `NumericSimplifier`, which should never be constructed
 * by users.
 */
class Polynomial {
 public:
  friend class Monomial;

  Polynomial() = default;
  Polynomial(const Polynomial& poly) = default;

  explicit Polynomial(const Monomial& m);

  Polynomial operator+() const;
  Polynomial operator-() const;

  Polynomial operator+(const Polynomial& b) const;
  Polynomial operator-(const Polynomial& b) const;
  Polynomial operator*(const Monomial& b) const;
  Polynomial operator*(const Polynomial& b) const;

  bool isMonomial() const;
  Monomial asMonomial() const;
  ExprPtr asExpr() const;

 private:
  void simpl();
  std::vector<Monomial> exprs_;
};

#define DECLR_PATTERN_SIMPLIFIER(SNAME)      \
  class SNAME : public PatternSimplifier {   \
   public:                                   \
    static NodePtr simplifyNode(Node* node); \
    SNAME();                                 \
  };

/**
 * @brief The simplifier for an expression with only one variable.
 *
 * This simplifier is used in monomial. If you would like to simplify an
 * expression with similiar patterns, please declare another simplifier and copy
 * the patterns, DO NOT use this simplifier directly.
 */
DECLR_PATTERN_SIMPLIFIER(SingleVarSimplifier);

/**
 * @brief Set VarPH to 1 in expression.
 */
DECLR_PATTERN_SIMPLIFIER(CoefficientGetter);

/**
 * @brief Simplify an expression before using `NumericSimplifier` and
 * `BoolSimplifier`.
 */
DECLR_PATTERN_SIMPLIFIER(PreSimplifier);

/**
 * @brief Simplify an expression before using `NumericSimplifier` and
 * `BoolSimplifier`.
 */
DECLR_PATTERN_SIMPLIFIER(PostSimplifier);

template <typename T>
NodePtr ExprSimplifier::visit(Array<T>* node) {
  auto res = std::make_shared<Array<T>>();
  for (auto& a : node->element)
    res->element.push_back(ptr_cast<T>(simplifyNode(a)));

  return res;
}

template <typename T>
void NumericSimplifier::visit(Const<T>* node) {
  push(Polynomial(Monomial(node->shared_from_this())));
}

#define CAST_TO(var, type, value)       \
  auto var = ir::ptr_cast<type>(value); \
  ELENA_ASSERT(var, "Cast failed to " #type);
#define EQ(a, b) (exprEqual(a.get(), b.get()))
#define NTYPE(node, type) (node->get_type() == IRNodeType::type)
#define MAX_CONST(a, b) (((a) >= (b)) ? (a) : (b))
#define MIN_CONST(a, b) (((a) >= (b)) ? (b) : (a))
#define ZERO PatternSimplifier::numWithType(0, dtype)
#define ONE PatternSimplifier::numWithType(1, dtype)
#define TWO PatternSimplifier::numWithType(2, dtype)
}  // namespace simplify
}  // namespace ir

#endif  // ELENA_INCLUDE_PASS_COMMON_SIMPLIFY_H_
