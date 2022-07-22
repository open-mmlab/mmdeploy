#include "IR/ExprEqual.h"

namespace ir {

bool constEqual(Expr* a, Expr* b) {
  ScalarType dtype = a->get_dtype();

  switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type: {                               \
    auto ca = static_cast<Const<native_type>*>(a);              \
    auto cb = static_cast<Const<native_type>*>(b);              \
    return ca->get_value() == cb->get_value();                  \
  }
#include "x/scalar_types.def"
  }
  return false;
}

bool exprEqual(Expr* a, Expr* b) {
#define REQUIRE(cond) \
  if (!(cond)) return false;
#define REQUIRE_END return true;
#define REQUIRE_FOR(type)           \
  auto* a_ = static_cast<type*>(a); \
  auto* b_ = static_cast<type*>(b);

  IRNodeType node_type = a->get_type();
  REQUIRE(node_type == b->get_type());
  switch (node_type) {
    case IRNodeType::Unary: {
      REQUIRE_FOR(Unary)
      REQUIRE(a_->operation_type == b_->operation_type)
      REQUIRE(exprEqual(a_->operand.get(), b_->operand.get()))
      REQUIRE_END
    }
    case IRNodeType::Binary: {
      REQUIRE_FOR(Binary)
      REQUIRE(a_->operation_type == b_->operation_type)
      REQUIRE(exprEqual(a_->lhs.get(), b_->lhs.get()))
      REQUIRE(exprEqual(a_->rhs.get(), b_->rhs.get()))
      REQUIRE_END
    }
    case IRNodeType::Logical: {
      REQUIRE_FOR(Logical)
      REQUIRE(a_->operation_type == b_->operation_type)
      REQUIRE(exprEqual(a_->lhs.get(), b_->lhs.get()))
      REQUIRE(exprEqual(a_->rhs.get(), b_->rhs.get()))
      REQUIRE_END
    }
    case IRNodeType::Select: {
      REQUIRE_FOR(ir::Select)
      REQUIRE(exprEqual(a_->cond.get(), b_->cond.get()))
      REQUIRE(exprEqual(a_->tBranch.get(), b_->tBranch.get()))
      REQUIRE(exprEqual(a_->fBranch.get(), b_->fBranch.get()))
      REQUIRE_END
    }
    case IRNodeType::Cast: {
      REQUIRE_FOR(Cast)
      REQUIRE(a_->get_dtype() == b_->get_dtype())
      REQUIRE(exprEqual(a_->expr_.get(), b_->expr_.get()))
      REQUIRE_END
    }
    case IRNodeType::IterVar: {
      REQUIRE_FOR(IterVar)
      REQUIRE(a_->get_name() == b_->get_name())
      REQUIRE_END
    }
    case IRNodeType::ScalarVar: {
      REQUIRE_FOR(ScalarVar)
      REQUIRE(a_->get_name() == b_->get_name())
      if (a_->tensor && a_->indices && b_->tensor && b_->indices) {
        REQUIRE(exprEqual(a_->indices->element[0].get(),
                          b_->indices->element[0].get()));
      }
      REQUIRE_END
    }
    case IRNodeType::Const: {
      REQUIRE(constEqual(a, b))
      REQUIRE_END
    }
    default:
      std::stringstream sstr;
      sstr << "Unknown Type" << std::endl;
      std::string str;
      sstr >> str;
      ELENA_ABORT(str.c_str());
  }

#undef REQUIRE_FOR
#undef REQUIRE_END
#undef REQUIRE
}
}  // namespace ir
