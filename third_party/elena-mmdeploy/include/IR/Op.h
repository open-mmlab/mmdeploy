#ifndef ELENA_INCLUDE_IR_OP_H_
#define ELENA_INCLUDE_IR_OP_H_

#include <cstddef>
#include <memory>
#include <string>

#include "IR/Container.h"
#include "IR/Expr.h"
#include "Stmt.h"
#include "Type.h"
#include "utils.h"

namespace ir {
class Expr;
class IterVar;
using ExprPtr = std::shared_ptr<Expr>;
class TensorVar;

/**
 * @brief Base class of operators.
 *
 * The output tensors stored in `Op::output` will have their `op` property
 * pointing to the operator itself.
 *
 * @author xupengcheng
 */
class Op : public Node, public std::enable_shared_from_this<Op> {
 public:
  virtual ~Op() = default;

  virtual TensorVarPtr output(ptrdiff_t i) = 0;
  virtual ArrayPtr<Expr> output_shape(ptrdiff_t i) = 0;
  virtual size_t output_count() const = 0;
  virtual std::string output_name() const = 0;

  static constexpr IRNodeType type = IRNodeType::Op;
  /** Expand the computation ability **/
  ExprPtr attached_intrin_expr = nullptr;

 protected:
  explicit Op(IRNodeType type);
};
/**
 * @brief Pointer to Op object.
 * @author xupengcheng
 */
using OpPtr = std::shared_ptr<Op>;

/**
 * @brief Base class of operators with singular output.
 * @warning Please do NOT use this in any interface.
 * @warning This is intended as an implementation detail only.
 * @author xieruifeng
 */
class SimpleOp : public Op {
 public:
  TensorVarPtr output(ptrdiff_t i) override;
  ArrayPtr<Expr> output_shape(ptrdiff_t i) override;
  size_t output_count() const override;
  std::string output_name() const override;

  using Op::type;

 protected:
  SimpleOp(ArrayPtr<Expr> shape, ScalarType dtype, std::string output_name,
           IRNodeType type);

  ArrayPtr<Expr> the_output_shape;
  ScalarType the_output_type;
  std::string the_output_name;
};

/**
 * @brief Operator for api::compute.
 * @author xupengcheng
 */
class ComputeOp : public SimpleOp {
  DECLARE_PRIVATE_CONSTRUCT

 public:
  /// Use the `create` static method to properly create a ComputeOp.
  ComputeOp(private_construct_t, ArrayPtr<Expr> shape,
            ArrayPtr<IterVar> iter_vars, ArrayPtr<IterVar> tensor_indices,
            ExprPtr f, std::string name);

  /**
   * @brief Create a ComputeOp properly.  The type of the output tensor for the
   * created ComputeOp is the same as f.
   * @param shape shape of the output tensor.
   * @param iter_vars iteration variables for compute expression, created by
   * api::construct_indices.
   * @return The constructed ComputeOp.
   * @author hanruobing
   */
  static OpPtr create(ArrayPtr<Expr> shape, ArrayPtr<IterVar> iter_vars,
                      ExprPtr f, std::string name);
  static OpPtr create(ArrayPtr<Expr> shape, ArrayPtr<IterVar> iter_vars,
                      ExprPtr f);
  static OpPtr create(ArrayPtr<Expr> shape, ArrayPtr<IterVar> for_iter_vars,
                      ArrayPtr<IterVar> tensor_indices, ExprPtr f,
                      std::string name);

  static constexpr IRNodeType type = IRNodeType::ComputeOp;

  ExprPtr fcompute;
  ArrayPtr<IterVar> iter_vars;
  /** Expand the computation ability **/
  ArrayPtr<IterVar> tensor_indices = nullptr;
};
/**
 * @brief Pointer to ComputeOp object.
 * @author xupengcheng
 */
using ComputeOpPtr = std::shared_ptr<ComputeOp>;

/**
 * @brief Operator for api::placeholder.
 * @author xupengcheng
 */
class PlaceholderOp : public SimpleOp {
  DECLARE_PRIVATE_CONSTRUCT
 public:
  /// Use the `create` static method to properly create a ComputeOp.
  explicit PlaceholderOp(private_construct_t, ArrayPtr<Expr> shape,
                         ScalarType dtype, std::string name);

  explicit PlaceholderOp(private_construct_t, ArrayPtr<Expr> shape,
                         ScalarType dtype);

  /**
   * @brief Create a PlaceholderOp properly.
   * @param shape shape of the output tensor.
   * @param dtype value type that the output tensor holds.
   * @return The constructed PlaceholderOp.
   */
  static OpPtr create(ArrayPtr<Expr> shape, ScalarType dtype, std::string name);
  static OpPtr create(ArrayPtr<Expr> shape, ScalarType dtype);

  static constexpr IRNodeType type = IRNodeType::PlaceholderOp;
};
/**
 * @brief Pointer to PlaceholderOp object.
 * @author xupengcheng
 */
using PlaceholderOpPtr = std::shared_ptr<PlaceholderOp>;

}  // namespace ir
#endif  // ELENA_INCLUDE_IR_OP_H_
