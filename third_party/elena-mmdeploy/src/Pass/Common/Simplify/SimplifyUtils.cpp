#include "IR/Expr.h"
#include "Pass/Common/Simplify.h"
#include "api.h"

namespace {

template <typename T>
ExprPtr makeConst(T value, ScalarType dtype) {
  return std::make_shared<Const<T>>(value, dtype);
}

}  // namespace

namespace ir {
namespace simplify {

// the number should be int
ExprPtr PatternSimplifier::numWithType(int val, ScalarType dtype) {
  switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type:                                 \
    return makeConst<native_type>(static_cast<native_type>(val), dtype);
#include "x/scalar_types.def"
    default:
      ELENA_ASSERT(false, "Unknown ScalarType");
  }
}

ExprPtr PatternSimplifier::eval(ExprPtr opd, UnaryType op) {
  auto dtype = opd->get_dtype();

  if (opd->get_type() != IRNodeType::Const) {
    return std::make_shared<Unary>(opd, op);
  }

  switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type: {                               \
    auto opd_c = ir::ptr_cast<Const<native_type>>(opd);         \
    native_type val = opd_c->get_value();                       \
                                                                \
    switch (op) {                                               \
      case UnaryType::Ceil:                                     \
        return makeConst<native_type>(std::ceil(val), dtype);   \
      default:                                                  \
        break;                                                  \
    }                                                           \
  }
#include "x/scalar_types.def"
  }

  return std::make_shared<Unary>(opd, op);
}

ExprPtr PatternSimplifier::eval(ExprPtr lhs, ExprPtr rhs, BinaryType op) {
  auto dtype = lhs->get_dtype();
  auto rdtype = rhs->get_dtype();

  if (lhs->get_type() != IRNodeType::Const ||
      rhs->get_type() != IRNodeType::Const) {
    return std::make_shared<Binary>(lhs, rhs, op);
  }

  switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type)               \
  case ScalarType::scalar_type: {                                             \
    auto lhs_c = ir::ptr_cast<Const<native_type>>(lhs);                       \
    native_type lval = lhs_c->get_value();                                    \
    native_type rval;                                                         \
    switch (rdtype) {                                                         \
      case ScalarType::Float32: {                                             \
        auto rhs_c = ir::ptr_cast<Const<float>>(rhs);                         \
        rval = static_cast<native_type>(rhs_c->get_value());                  \
        break;                                                                \
      }                                                                       \
      default: {                                                              \
        auto rhs_c = ir::ptr_cast<Const<native_type>>(rhs);                   \
        rval = static_cast<native_type>(rhs_c->get_value());                  \
        break;                                                                \
      }                                                                       \
    }                                                                         \
    switch (op) {                                                             \
      case BinaryType::Add:                                                   \
        return makeConst<native_type>(lval + rval, dtype);                    \
      case BinaryType::Sub:                                                   \
        if ((dtype == ScalarType::UInt64 || dtype == ScalarType::UInt32 ||    \
             dtype == ScalarType::UInt8) &&                                   \
            lval <= rval) {                                                   \
          return std::make_shared<Unary>(                                     \
              makeConst<native_type>(rval - lval, dtype), UnaryType::Negate); \
        }                                                                     \
        return makeConst<native_type>(lval - rval, dtype);                    \
      case BinaryType::Mul:                                                   \
        return makeConst<native_type>(lval * rval, dtype);                    \
      case BinaryType::Div:                                                   \
        return makeConst<native_type>(lval / rval, dtype);                    \
      case BinaryType::Mod:                                                   \
        return makeConst<native_type>(lval % rval, dtype);                    \
      case BinaryType::And:                                                   \
        return makeConst<native_type>(lval && rval, dtype);                   \
      case BinaryType::Or:                                                    \
        return makeConst<native_type>(lval || rval, dtype);                   \
      case BinaryType::Max:                                                   \
        return makeConst<native_type>(MAX_CONST(lval, rval), dtype);          \
      case BinaryType::Min:                                                   \
        return makeConst<native_type>(MIN_CONST(lval, rval), dtype);          \
      default:                                                                \
        break;                                                                \
    }                                                                         \
  }

#define TYPE_MAP_NATIVE_TO_SCALARTYPE_FP(native_type, scalar_type)   \
  case ScalarType::scalar_type: {                                    \
    auto lhs_c = ir::ptr_cast<Const<native_type>>(lhs);              \
    auto rhs_c = ir::ptr_cast<Const<native_type>>(rhs);              \
                                                                     \
    native_type lval = lhs_c->get_value();                           \
    native_type rval = rhs_c->get_value();                           \
                                                                     \
    switch (op) {                                                    \
      case BinaryType::Add:                                          \
        return makeConst<native_type>(lval + rval, dtype);           \
      case BinaryType::Sub:                                          \
        return makeConst<native_type>(lval - rval, dtype);           \
      case BinaryType::Mul:                                          \
        return makeConst<native_type>(lval * rval, dtype);           \
      case BinaryType::Div:                                          \
        return makeConst<native_type>(lval / rval, dtype);           \
      case BinaryType::And:                                          \
        return makeConst<native_type>(lval && rval, dtype);          \
      case BinaryType::Or:                                           \
        return makeConst<native_type>(lval || rval, dtype);          \
      case BinaryType::Max:                                          \
        return makeConst<native_type>(MAX_CONST(lval, rval), dtype); \
      case BinaryType::Min:                                          \
        return makeConst<native_type>(MIN_CONST(lval, rval), dtype); \
      default:                                                       \
        break;                                                       \
    }                                                                \
  }
#include "x/scalar_types.def"
  }
  return nullptr;
}

ExprPtr PatternSimplifier::eval(ExprPtr expr, ScalarType dst_type) {
  ScalarType src_type = expr->get_dtype();

  if (expr->get_type() != IRNodeType::Const) return expr;
  if (dst_type == src_type) return expr;

#define CAST_TYPE_MAP(src_ntype, dst_ntype, src_dtype, dst_dtype)            \
  if (src_type == ScalarType::src_dtype &&                                   \
      dst_type == ScalarType::dst_dtype) {                                   \
    auto c = ptr_cast<Const<src_ntype>>(expr);                               \
    return api::constant<dst_ntype>(static_cast<dst_ntype>(c->get_value())); \
  }
#include "x/cast_types.def"

  ELENA_ASSERT(false, "Cannot cast the type");
  return nullptr;
}

}  // namespace simplify
}  // namespace ir
