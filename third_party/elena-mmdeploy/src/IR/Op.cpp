#include "IR/Op.h"

#include "IR/Stage.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

namespace ir {

Op::Op(IRNodeType type) : Node(type) {}

SimpleOp::SimpleOp(ArrayPtr<Expr> shape, ScalarType dtype,
                   std::string output_name, IRNodeType type)
    : Op(type),
      the_output_shape(shape),
      the_output_type(dtype),
      the_output_name(output_name) {}

TensorVarPtr SimpleOp::output(ptrdiff_t i) {
  ELENA_ASSERT(i == 0, "SimpleOp has only one output.");
  return std::make_shared<TensorVar>(the_output_name, the_output_shape,
                                     shared_from_this(), the_output_type);
}

ArrayPtr<Expr> SimpleOp::output_shape(ptrdiff_t i) { return the_output_shape; }

std::string SimpleOp::output_name() const { return the_output_name; }

size_t SimpleOp::output_count() const { return 1; }

DEFINE_PRIVATE_CONSTRUCT(ComputeOp)

ComputeOp::ComputeOp(private_construct_t, ArrayPtr<Expr> shape,
                     ArrayPtr<IterVar> iter_vars,
                     ArrayPtr<IterVar> tensor_indices, ExprPtr f,
                     std::string name)
    : SimpleOp(shape, f->get_dtype(), name, type),
      fcompute(f),
      iter_vars(iter_vars),
      tensor_indices(tensor_indices) {
  for (const auto& a : shape->element) {
    CHECK_DATA_TYPE(a, UInt64)
  }
  for (const auto& a : iter_vars->element) {
    CHECK_DATA_TYPE(a, UInt64)
  }
}

OpPtr ComputeOp::create(ArrayPtr<Expr> shape, ArrayPtr<IterVar> iter_vars,
                        ExprPtr f) {
  return std::make_shared<ComputeOp>(private_construct, shape, iter_vars,
                                     nullptr, f, GENERATE_NAME(ComputeOp));
}

OpPtr ComputeOp::create(ArrayPtr<Expr> shape, ArrayPtr<IterVar> iter_vars,
                        ExprPtr f, std::string name) {
  return std::make_shared<ComputeOp>(private_construct, shape, iter_vars,
                                     nullptr, f, name);
}

OpPtr ComputeOp::create(ArrayPtr<Expr> shape, ArrayPtr<IterVar> for_iter_vars,
                        ArrayPtr<IterVar> tensor_indices, ExprPtr f,
                        std::string name) {
  return std::make_shared<ComputeOp>(private_construct, shape, for_iter_vars,
                                     tensor_indices, f, name);
}

DEFINE_PRIVATE_CONSTRUCT(PlaceholderOp)

PlaceholderOp::PlaceholderOp(private_construct_t, ArrayPtr<Expr> shape,
                             ScalarType dtype, std::string name)
    : SimpleOp(shape, dtype, name, type) {
  for (const auto& a : shape->element) {
    CHECK_DATA_TYPE(a, UInt64)
  }
}

PlaceholderOp::PlaceholderOp(private_construct_t, ArrayPtr<Expr> shape,
                             ScalarType dtype)
    : SimpleOp(shape, dtype, GENERATE_NAME(PlaceholderOp), type) {
  for (const auto& a : shape->element) {
    CHECK_DATA_TYPE(a, UInt64)
  }
}

OpPtr PlaceholderOp::create(ArrayPtr<Expr> shape, ScalarType dtype,
                            std::string name) {
  return std::make_shared<PlaceholderOp>(private_construct, shape, dtype, name);
}

OpPtr PlaceholderOp::create(ArrayPtr<Expr> shape, ScalarType dtype) {
  return std::make_shared<PlaceholderOp>(private_construct, shape, dtype,
                                         GENERATE_NAME(PlaceholderOp));
}
}  // namespace ir
