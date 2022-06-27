#include "IR/IntSet.h"

#include <vector>

#include "api.h"
#include "logging.h"

namespace {
bool isConst(const ExprPtr& expr) {
  return expr->get_type() == ir::IRNodeType::Const;
}

uint64_t getConstVal(const ExprPtr& expr) {
  ELENA_ASSERT_EQ(expr->get_dtype(), ir::ScalarType::UInt64,
                  "Error expression type");
  auto const_ptr = ir::ptr_cast<ir::Const<uint64_t>, ir::Expr>(expr);
  ELENA_ASSERT(const_ptr, "Not const expression");
  return const_ptr->get_value();
}

#define BUILD_EXPR_EVAL(res, op, lhs, rhs)                                 \
  {                                                                        \
    if (isConst(lhs) && isConst(rhs)) {                                    \
      res = api::constant<uint64_t>(getConstVal(lhs) op getConstVal(rhs)); \
    } else if ((lhs == rhs) && (#op == std::string("-"))) {                \
      res = api::constant<uint64_t>(0);                                    \
    } else {                                                               \
      res = lhs op rhs;                                                    \
    }                                                                      \
  }
}  // namespace

IntSet::IntSet() { setInf(); }
IntSet::IntSet(const ir::RangePtr& r) { setRange(r); }
IntSet::IntSet(const ir::ExprPtr min, const ir::ExprPtr max)
    : min_value(min), max_value(max) {}

void IntSet::setRange(const ir::RangePtr& r) {
  ExprPtr ext;
  BUILD_EXPR_EVAL(ext, -, r->extent, api::constant<uint64_t>(1))
  if (r->stride) {  // has stride field
    BUILD_EXPR_EVAL(ext, *, ext, r->stride);
  }

  min_value = r->init;
  BUILD_EXPR_EVAL(max_value, +, min_value, ext);
}

void IntSet::setInf() {
  min_value = api::constant<uint64_t>(0);
  max_value = api::constant<uint64_t>(std::numeric_limits<uint64_t>::max());
}

void IntSet::setSinglePoint(const ir::ExprPtr& expr) {
  min_value = max_value = expr;
}

ir::RangePtr IntSet::getRange() const {
  ExprPtr init = min_value, extent, stride;
  BUILD_EXPR_EVAL(extent, -, max_value, min_value);
  BUILD_EXPR_EVAL(extent, +, extent, api::constant<uint64_t>(1))
  stride = api::constant<uint64_t>(1);
  return std::make_shared<ir::Range>(init, extent, stride);
}

bool IntSet::isEmpty() const { return min_value == nullptr; }

bool IntSet::isSinglePoint() const { return min_value == max_value; }

ir::ExprPtr IntSet::min() const { return min_value; }

ir::ExprPtr IntSet::max() const { return max_value; }

IntSet IntSet::merge(const IntSet& s) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;

  if (isConst(min_value) && isConst(max_value) && isConst(s.min_value) &&
      isConst(s.max_value)) {
    is.min_value = api::constant<uint64_t>(
        std::min(getConstVal(min_value), getConstVal(s.min_value)));
    is.max_value = api::constant<uint64_t>(
        std::max(getConstVal(max_value), getConstVal(s.max_value)));
    return is;
  }
  std::string msg;
  msg =
      "[USE API::MAX/MIN INSTEAD]trickyimplement for merge() which is not "
      "const.\n";
  ELENA_LOG_INFO(msg.c_str());
  is.min_value = min_value;
  is.max_value = max_value;
  return is;

  // is.min_value =
  //     std::make_shared<ir::Binary>(min_value,
  //                                  s.min_value,
  //                                  ir::BinaryType::min);
  // is.max_value =
  //     std::make_shared<ir::Binary>(max_value,
  //                                  s.max_value, ir::BinaryType::max);
  // is.setInf();
  auto min_value_diff_from = api::simplify(min_value - s.min_value);
  auto min_value_diff_to = api::simplify(s.min_value - min_value);
  if (isConst(min_value_diff_from)) {
    is.min_value = s.min_value;
  } else if (isConst(min_value_diff_to)) {
    is.min_value = min_value;
  } else {
    ELENA_ABORT("Cannot convert intset min value to const")
  }
  auto max_value_diff_from = api::simplify(max_value - s.max_value);
  auto max_value_diff_to = api::simplify(s.max_value - max_value);

  if (isConst(max_value_diff_from)) {
    is.max_value = max_value;
  } else if (isConst(max_value_diff_to)) {
    is.max_value = s.min_value;
  } else {
    ELENA_ABORT("Cannot convert intset max value to const")
  }
  return is;
}

IntSet IntSet::operator+(const IntSet& s) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, +, min_value, s.min_value);
  BUILD_EXPR_EVAL(is.max_value, +, max_value, s.max_value);
  return is;
}

IntSet IntSet::operator/(const IntSet& s) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, /, min_value, s.min_value);
  BUILD_EXPR_EVAL(is.max_value, /, max_value, s.max_value);
  return is;
}

IntSet IntSet::operator%(const IntSet& s) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, %, min_value, s.min_value);
  BUILD_EXPR_EVAL(is.max_value, %, max_value, s.max_value);
  return is;
}

IntSet IntSet::operator*(const IntSet& s) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, *, min_value, s.min_value);
  BUILD_EXPR_EVAL(is.max_value, *, max_value, s.max_value);
  return is;
}

IntSet IntSet::operator-(const IntSet& s) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, -, min_value, s.min_value);
  BUILD_EXPR_EVAL(is.max_value, -, max_value, s.max_value);
  return is;
}

IntSet IntSet::operator+(const ir::ExprPtr& expr) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, +, min_value, expr);
  BUILD_EXPR_EVAL(is.max_value, +, max_value, expr);
  return is;
}

IntSet IntSet::operator*(const ir::ExprPtr& expr) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, *, min_value, expr);
  BUILD_EXPR_EVAL(is.max_value, *, max_value, expr);
  return is;
}

IntSet IntSet::operator/(const ir::ExprPtr& expr) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, /, min_value, expr);
  BUILD_EXPR_EVAL(is.max_value, /, max_value, expr);
  return is;
}

IntSet IntSet::operator%(const ir::ExprPtr& expr) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, %, min_value, expr);
  BUILD_EXPR_EVAL(is.max_value, %, max_value, expr);
  return is;
}

IntSet IntSet::operator-(const ir::ExprPtr& expr) const {
  ELENA_ASSERT(min_value, "Empty range in IntSet");
  IntSet is;
  BUILD_EXPR_EVAL(is.min_value, -, min_value, expr);
  BUILD_EXPR_EVAL(is.max_value, -, max_value, expr);
  return is;
}

IntSet merge(const IntSet& ls, const IntSet& rs) { return ls.merge(rs); }

IntSet merge(std::vector<IntSetPtr>* IntSet_vector) {
  /*if (IntSet_vector->size() == 0) {
    return IntSet();
  } else*/
  if (IntSet_vector->size() == 1) {
    return *((*IntSet_vector)[0]);
  } else {
    IntSet result = *((*IntSet_vector)[0]);
    for (int i = 1; i < IntSet_vector->size(); i++) {
      IntSet value = *((*IntSet_vector)[i]);
      result = merge(result, value);
    }
    return result;
  }
}

IntSet ceil(const IntSet& s) {
  ExprPtr max_expr;
  BUILD_EXPR_EVAL(max_expr, +, s.max(), api::constant<uint64_t>(1));
  IntSet is(s.min(), max_expr);
  return is;
}

/// Revaluate the range of IterVars with the newest values.
class EvalIterVarRange final : public VisitorBase<EvalIterVarRange> {
 private:
  /// variable to store the intermediate value.
  ExprPtr mid_result;

  /// The Map used for the revaluation.
  const Rmap* up_state;
  /// Shows if the revaluation is operated on the minmal boundary.
  bool is_min;

 public:
  using VisitorBase::visit;

  /// Visit instances of class IterVar.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(iter_var);
  /// \encode
  ///
  /// \param iter_ptr instance of class IterVar;
  ///
  /// \return None.
  void visit(ir::IterVar* iter_ptr) {
    const auto node = iter_ptr->shared_from_this();
    // set range if itervar can be found in Rmap and the range is const
    if (up_state->find(node) != up_state->end()) {
      auto iter_is = up_state->find(node)->second;
      // if (iter_is->min()->get_type() == ir::IRNodeType::Const)
      mid_result = (is_min ? iter_is->min() : iter_is->max());
      return;
    }

    std::stringstream sstr;
    sstr << iter_ptr->get_name() << " not found.\n";
    std::string str;
    sstr >> str;
    ELENA_WARN(str.c_str());
    mid_result = ir::ptr_cast<ir::Expr>(node);
  }

  /// Visit instances of class Const.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(const);
  /// \encode
  ///
  /// \param node_ptr instance of class Const;
  ///
  /// \return None.
  template <typename T>
  void visit(ir::Const<T>* node) {
    mid_result = ir::ptr_cast<ir::Expr>(node->shared_from_this());
  }

  /// Visit instances of class ScalarVar.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(scalarvar);
  /// \encode
  ///
  /// \param node_ptr instance of class ScalarVar;
  ///
  /// \return None.
  void visit(ir::ScalarVar* node) {
    mid_result = node->shared_from_this();
  }

  /// Visit instances of class Binary.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(binary_ptr);
  /// \encode
  ///
  /// \param binary_ptr instance of class Binary;
  ///
  /// \return None.
  void visit(ir::Binary* binary_ptr) {
    visit(binary_ptr->lhs.get());
    ir::ExprPtr lhs = mid_result;
    visit(binary_ptr->rhs.get());
    ir::ExprPtr rhs = mid_result;
    switch (binary_ptr->operation_type) {
      case ir::BinaryType::Add:
        mid_result = lhs + rhs;
        break;
      case ir::BinaryType::Mul:
        mid_result = lhs * rhs;
        break;
      case ir::BinaryType::Sub:
        mid_result = lhs - rhs;
        break;
      case ir::BinaryType::Div:
        mid_result = lhs / rhs;
        break;
      case ir::BinaryType::Mod:
        mid_result = lhs % rhs;
        break;
      default:
        ELENA_ABORT("Binary type"
                    << BINARYTYPE_SYMBOL(binary_ptr->operation_type)
                    << " is not supported.");
    }
  }

  /// Visit instances of class Unary.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(unary_ptr);
  /// \encode
  ///
  /// \param unary_ptr instance of class Unary;
  ///
  /// \return None.
  void visit(ir::Unary* unary_ptr) {
    visit(unary_ptr->operand.get());
    ir::ExprPtr opr = mid_result;
    switch (unary_ptr->operation_type) {
      case ir::UnaryType::Negate:
        mid_result = -opr;
        break;
      case ir::UnaryType::Floor:
        // Consider statement simplify, now do not deal with the floor
        break;
      case ir::UnaryType::Ceil:
        mid_result = std::make_shared<ir::Cast>(
            std::make_shared<ir::Unary>(opr, ir::UnaryType::Ceil),
            ir::ScalarType::UInt64);
        break;
      default:
        return;
        ELENA_ABORT("Unary type " << UNARYTYPE_SYMBOL(unary_ptr->operation_type)
                                  << " is not supported.");
    }
  }

  /// Revaluate the value of node.
  ///
  /// Typical Usage:
  /// \code
  ///   getExpr(node, rmap, true);
  /// \encode
  ///
  /// \param up_state_ the map which records the range of IterVars;
  /// \param is_min_ shows if the revaluated value is the
  /// minimal boundary;
  ///
  /// \return None.
  ir::ExprPtr getExpr(const ir::NodePtr& node, const Rmap& up_state_,
                      bool is_min_) {
    up_state = &up_state_;
    mid_result = nullptr;
    is_min = is_min_;
    visit(node.get());
    return mid_result;
  }
};

void IntSet::evalItervarRange(const Rmap& r) {
  EvalIterVarRange eir;
  min_value = eir.getExpr(api::simplify(min_value), r, true);
  max_value = eir.getExpr(api::simplify(max_value), r, false);
}

void evalAllItervarRange(Rmap& r) {  // NOLINT
  for (auto& i : r) {
    auto var = i.first;
    IntSetPtr& is = i.second;
    is->evalItervarRange(r);
  }
}
