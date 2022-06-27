#ifndef ELENA_INCLUDE_IR_EXPR_H_
#define ELENA_INCLUDE_IR_EXPR_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stack>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "IR/Container.h"
#include "IR/NameGenerator.h"
#include "Node.h"
#include "Type.h"
#include "logging.h"
#include "utils.h"

/**
 * @brief Namespace for all IR nodes.
 * @author xupengcheng
 */
namespace ir {

class Range;

/**
 * @brief This is the base class of all expressions.
 *
 * All Parser IR nodes are subclasses of this class.  Most of the operations
 * represented by these nodes are purely functional.
 *
 * @author xupengcheng
 */
class Expr : public Node {
 public:
  static constexpr IRNodeType type = IRNodeType::Expr;
  explicit Expr(ScalarType dtype, IRNodeType type = Expr::type);

  ScalarType get_dtype() const;
  void set_dtype(ScalarType);
  virtual ~Expr();

 private:
  ScalarType dtype;
};
/**
 * @brief Pointer to Expr object.
 * @author xupengcheng
 */
using ExprPtr = std::shared_ptr<Expr>;

/**
 * @brief operator+ overloads for Expr.  Creates new Binary node that
 * represents add.
 * @author xupengcheng
 */
ExprPtr operator+(ExprPtr lhs, ExprPtr rhs);
/**
 * @brief operator- overloads for Expr.  Creates new Binary node that
 * represents subtract.
 * @author xupengcheng
 */
ExprPtr operator-(ExprPtr lhs, ExprPtr rhs);
/**
 * @brief operator* overloads for Expr.  Creates new Binary node that
 * represents multiply.
 * @author xupengcheng
 */
ExprPtr operator*(ExprPtr lhs, ExprPtr rhs);
/**
 * @brief operator/ overloads for Expr.  Creates new Binary node that
 * represents divide.
 * @author xupengcheng
 */
ExprPtr operator/(ExprPtr lhs, ExprPtr rhs);
/**
 * @brief operator% overloads for Expr.  Creates new Binary node that
 * represents modulo.
 * @author xupengcheng
 */
ExprPtr operator%(ExprPtr lhs, ExprPtr rhs);
/**
 * @brief operator- overloads for Expr.  Creates new Unary node that
 * represents negate.
 * @author xupengcheng
 */
ExprPtr operator-(ExprPtr operand);

/**
 * @brief operator^ overloads for Expr.  Creates new Binary node that
 * represents pow.
 * @author xiaoquanlun
 */
ExprPtr operator^(ExprPtr lhs, ExprPtr rhs);

ExprPtr same(ExprPtr lhs, ExprPtr rhs);
ExprPtr lt(ExprPtr lhs, ExprPtr rhs);
ExprPtr gt(ExprPtr lhs, ExprPtr rhs);
ExprPtr le(ExprPtr lhs, ExprPtr rhs);
ExprPtr ge(ExprPtr lhs, ExprPtr rhs);
ExprPtr and_(ExprPtr lhs, ExprPtr rhs);
ExprPtr or_(ExprPtr lhs, ExprPtr rhs);
ExprPtr expr_and(ExprPtr lhs, ExprPtr rhs);
ExprPtr expr_or(ExprPtr lhs, ExprPtr rhs);

/**
 * @brief operator max/min overloads for Expr.  Creates new Binary node that
 * represents max/min.
 * @author xiaoquanlun
 */

ExprPtr max(ExprPtr lhs, ExprPtr rhs);

ExprPtr min(ExprPtr lhs, ExprPtr rhs);

#define TYPE_UNARYTYPE_FUNC_OP_MAP(BIG_ID, SMALL_ID) \
  ExprPtr SMALL_ID(ExprPtr lhs);
#include "x/unary_types.def"

/**
 * @brief Node for binary operations.
 *
 * Binary assumes that both hands are of the same data type; this is enforced by
 * the constructor.  The detailed computation is denoted by the BinaryType enum.
 *
 * @author xupengcheng
 */
class Binary : public Expr, public std::enable_shared_from_this<Binary> {
 public:
  static constexpr IRNodeType type = IRNodeType::Binary;
  Binary(ExprPtr l, ExprPtr r, BinaryType binary_type);

  ExprPtr lhs;
  ExprPtr rhs;
  BinaryType operation_type;
};

/**
 * @brief Pointer to Binary object.
 * @author xupengcheng
 */
using BinaryPtr = std::shared_ptr<Binary>;

/**
 * @brief Node for unary operations.
 *
 * The detailed computation is denoted by the UnaryType enum.
 *
 * @author xupengcheng
 */
class Unary : public Expr, public std::enable_shared_from_this<Unary> {
 public:
  static constexpr IRNodeType type = IRNodeType::Unary;
  Unary(ExprPtr value, UnaryType unary_type);

  ExprPtr operand;
  UnaryType operation_type;
};
/**
 * @brief Pointer to Unary object.
 * @author xupengcheng
 */
using UnaryPtr = std::shared_ptr<Unary>;

/**
 * @brief Logical operation for comparison between scalar expressions.
 *
 * Logical assumes that both hands are of type ScalarType::Boolean; this is
 * enforced by the constructor.  The detailed computation is denoted by the
 * LogicalType enum.
 *
 * @author xupengcheng
 */
class Logical : public Expr, public std::enable_shared_from_this<Logical> {
 public:
  static constexpr IRNodeType type = IRNodeType::Logical;
  Logical(ExprPtr l, ExprPtr r, LogicalType logical_type);

  ExprPtr lhs;
  ExprPtr rhs;
  LogicalType operation_type;
};
/**
 * @brief Pointer to Logical object.
 * @author xupengcheng
 */
using LogicalPtr = std::shared_ptr<Logical>;

/**
 * @brief ternary operator 'cond ? tBranch : fBranch'.
 * @author xieruifeng
 */
class Select : public Expr, public std::enable_shared_from_this<Select> {
 public:
  static constexpr IRNodeType type = IRNodeType::Select;
  Select(ExprPtr cond, ExprPtr tBranch, ExprPtr fBranch);

  ExprPtr cond;
  ExprPtr tBranch;
  ExprPtr fBranch;
};
/**
 * @brief Pointer to Logical object.
 * @author xupengcheng
 */
using SelectPtr = std::shared_ptr<Select>;

/**
 * @brief Node for external function call that returns a scalar.
 *
 * Note that the function name may be matched for intrinsic functions, such as
 * the *select* function, which is used to implement api::if_then_else.
 *
 * @author xupengcheng
 */
class Call : public Expr, public std::enable_shared_from_this<Call> {
 public:
  static constexpr IRNodeType type = IRNodeType::Call;
  explicit Call(CallFunction func, ArrayPtr<Expr> args, ScalarType dtype);

  CallFunction func;
  ArrayPtr<Expr> args;
};
/**
 * @brief Pointer to Call object.
 * @author xupengcheng
 */
using CallPtr = std::shared_ptr<Call>;

/**
 * @brief Node for constant scalar value.
 *
 * This class uses the same nested type technique as Array (with
 * NestedTypeNode), while the nested type is ScalarType.
 *
 * @author xupengcheng
 */
template <typename T>
class Const : public Expr, public std::enable_shared_from_this<Const<T>> {
 public:
  static constexpr IRNodeType type = IRNodeType::Const;
  Const(T value, ScalarType dtype);

  virtual ~Const();
  T get_value() const;
  std::string get_name() const;

 private:
  T value;
  std::string name;
};
template <typename T>
/**
 * @brief Pointer to Const object.
 * @author xupengcheng
 */
using ConstPtr = std::shared_ptr<Const<T>>;

/**
 * @brief Node for a string label.
 *
 * Mainly used in attribute nodes.
 *
 * @author xupengcheng
 */
class Label : public Node, public std::enable_shared_from_this<Label> {
 public:
  static constexpr IRNodeType type = IRNodeType::Label;
  explicit Label(const std::string &v);

  const std::string &get_value() const;

 private:
  std::string value;
};
/**
 * @brief Pointer to Label object.
 * @author xupengcheng
 */
using LabelPtr = std::shared_ptr<Label>;

/**
 * @brief Node for a variable.
 *
 * This is the general variable class; no objects should be of this type
 * directly.
 *
 * @author xupengcheng
 */
class Var : public Expr {
 public:
  static constexpr IRNodeType type = IRNodeType::Var;
  explicit Var(ScalarType dtype, IRNodeType type = Var::type);
  virtual ~Var();

  virtual const std::string &get_name() const = 0;
  virtual void set_name(const std::string &name) = 0;
};
/**
 * @brief Pointer to Var object.
 * @author xupengcheng
 */
using VarPtr = std::shared_ptr<Var>;

/**
 * @brief Variables that take care of its own name.
 * @warning Please do NOT use this in any interface.
 * @warning This is intended as an implementation detail only.
 * @author xieruifeng
 */
class NamedVar : public Var {
 public:
  NamedVar(std::string name, ScalarType dtype, IRNodeType type = Var::type);

  const std::string &get_name() const override;
  void set_name(const std::string &name) override;

 private:
  std::string var_name;
};

using NamedVarPtr = std::shared_ptr<NamedVar>;

class TensorVar;

/**
 * @brief Pointer to TensorVar object.
 * @author xupengcheng
 */
using TensorVarPtr = std::shared_ptr<TensorVar>;

/**
 * @brief Node for a scalar variable.
 *
 * The scalar can come from an api::var call or a read from a TensorVar.
 *
 * @author xupengcheng
 */
class ScalarVar : public NamedVar,
                  public std::enable_shared_from_this<ScalarVar> {
 public:
  static constexpr IRNodeType type = IRNodeType::ScalarVar;
  /**
   * @brief Construct a ScalarVar from api::var.  The tensor and indices fields
   * are default-initialized (nullptr).
   * @author xupengcheng
   * @param name name of the variable.
   * @param dtype data type of the variable.
   */
  explicit ScalarVar(const std::string &name, ScalarType dtype);
  explicit ScalarVar(ScalarType dtype);

  /**
   * @brief Construct a ScalarVar from subscripting a TensorVar
   * (TensorVar::operator()).  The tensor and indices fields are filled
   * accordingly.
   * @author xupengcheng
   * @param tensor the tensor from which this scalar is taken.
   * @param indices the indices of this scalar in the tensor it belongs to.
   * @param name name of the variable.
   */
  ScalarVar(ExprPtr tensor, ArrayPtr<Expr> indices, const std::string &name);
  ScalarVar(ExprPtr tensor, ArrayPtr<Expr> indices);

  /**
   * @brief Checks if the ScalarVar is a placeholder (from api::var).
   * Placeholder variables won't get their tensor / indices fields visited in
   * VisitorBase.
   * @author xupengcheng
   * @return if the variable is a placeholder.
   */
  bool is_placeholder() const;

  TensorVarPtr tensor = nullptr;
  ArrayPtr<Expr> indices = nullptr;
};
/**
 * @brief Pointer to ScalarVar object.
 * @author xupengcheng
 */
using ScalarVarPtr = std::shared_ptr<ScalarVar>;

/**
 * @brief Node for scalar assignment.
 *
 * ScalarAssign assumes that both hands are of the same data type; this is enforced by
 * the constructor. Besides, values should be scalar expressions.
 *
 * @author jianglijuan
 */
class ScalarAssign : public Expr,
    public std::enable_shared_from_this<ScalarAssign> {
 public:
  static constexpr IRNodeType type = IRNodeType::ScalarAssign;
  ScalarAssign(ScalarVarPtr var, ExprPtr value);

  ScalarVarPtr var;
  ExprPtr value;
};

using ScalarAssignPtr = std::shared_ptr<ScalarAssign>;

class Op;
using OpPtr = std::shared_ptr<Op>;

/**
 * @brief Node for a tensor key.
 *
 * @details used in flattenStorage, record one switched TensorVar with
 * TensorKey, and change it when used later because it was already changed when
 * allocated
 *
 * @author xiaoquanlun
 */

class TensorKey {
 public:
  TensorKey(std::string,
            ArrayPtr<Range>);  // tensor_name and original tensor_shape
  std::string get_key();
  ArrayPtr<Range> get_bound();
  /*inline bool operator==(const TensorKey& other) const {
  return key == other.key && bound == other.bound;
  }*/
 private:
  ArrayPtr<Range> bound;
  std::string key;
};

using TensorKeyPtr = std::shared_ptr<TensorKey>;

/**
 * @brief Node for a tensor variable.
 *
 * The tensor can come from an api::placeholder call or api::compute.
 *
 * @author xupengcheng
 */
class TensorVar : public Var, public std::enable_shared_from_this<TensorVar> {
 protected:
  template <typename T>
  std::shared_ptr<T> shared_from(T *derived) {
    assert(this == derived);
    return std::static_pointer_cast<T>(shared_from_this());
  }

 public:
  struct hash {
    using argument_type = TensorVarPtr;
    using result_type = std::size_t;
    std::size_t operator()(const TensorVarPtr &p) const {
      // Hopefully no valid tensor has an empty name.
      return std::hash<std::string>()(p ? p->get_name() : "");
    }
  };

  struct equality {
    using result_type = bool;
    using first_argument_type = TensorVarPtr;
    using second_argument_type = TensorVarPtr;
    bool operator()(const TensorVarPtr &lhs, const TensorVarPtr &rhs) const {
      if (!lhs && !rhs) return true;
      if (!lhs || !rhs) return false;
      return lhs->get_name() == rhs->get_name();
    }
  };

  TensorVar(std::shared_ptr<std::string> name, ArrayPtr<Expr> shape, OpPtr op,
            ScalarType dtype);
  TensorVar(ArrayPtr<Expr> shape, OpPtr op, ScalarType dtype);

  static constexpr IRNodeType type = IRNodeType::TensorVar;
  TensorVar(std::string &name, ArrayPtr<Expr> shape, OpPtr op,
            ScalarType dtype);

  const std::string &get_name() const override;
  void set_name(const std::string &name) override;

  /**
   * @brief Subscription operator for TensorVar.  Note that TensorVarPtr is
   * usually used to pass the TensorVar around; dereference first before calling
   * `operator()`.   Note that the number of arguments should match the
   * dimensions of the tensor, or an assertion will fail.  Example usage:
   * ```
   * auto n = api::var("n");
   * auto two = api::constant<int64_t>(2);
   * auto tensor = api::placeholder({n}, "tensor");
   * auto scalar = (*tensor)(n / two);
   * ```
   * @author xupengcheng
   * @param indices Coordinates for the desired value.
   * @return ExprPtr pointer to a ScalarVar for the location.
   */
  template <typename... Args>
  ExprPtr operator()(Args... indices) {
    ELENA_ASSERT_EQ(sizeof...(Args), shape->element.size(),
                    "Subscript indices should match tensor dimensions");
    return std::make_shared<ScalarVar>(
        shared_from_this(),
        std::make_shared<Array<Expr>>(
            std::initializer_list<ExprPtr>({indices...})),
        get_name() + "_slice");
  }

  ExprPtr operator()(std::vector<ExprPtr> indices);

  Array<Expr> &get_shape();

  OpPtr get_op();

  ArrayPtr<Expr> shape;
  OpPtr op;

 private:
  std::shared_ptr<std::string> var_name;
};

template <typename T>
using TensorVarMap =
    std::unordered_map<TensorVarPtr, T, TensorVar::hash, TensorVar::equality>;

/**
 * @brief Node for a range.
 * @author guanzhichao, lixiuhong
 *
 */
class Range : public Node, public std::enable_shared_from_this<Range> {
 public:
  static constexpr IRNodeType type = IRNodeType::Range;

  Range();
  Range(ExprPtr init, ExprPtr extent);
  Range(ExprPtr init, ExprPtr extent, ExprPtr stride);
  explicit Range(const std::shared_ptr<Range> &srange);

  ExprPtr init;
  ExprPtr extent;
  ExprPtr stride;

  bool is_null();
};

using RangePtr = std::shared_ptr<Range>;
using Region = Array<Range>;
using RegionPtr = std::shared_ptr<Region>;

/**
 * @brief Node for an iteration variable.
 * @author xupengcheng
 */
class IterVar : public NamedVar, public std::enable_shared_from_this<IterVar> {
 public:
  static constexpr IRNodeType type = IRNodeType::IterVar;
  IterVar(ExprPtr init, ExprPtr extent, const std::string &name,
          bool is_reduce_ = false);
  IterVar(RangePtr range, const std::string &name, bool is_reduce_ = false);

  RangePtr range;

  IterAttrType iter_type;
  std::string thread_tag;
  bool is_reduce;
  // always make use of all itervar range no matter of its comsumers' range
  bool fullfill_range = false;
};
/**
 * @brief Pointer to IterVar object.
 * @author xupengcheng
 */
using IterVarPtr = std::shared_ptr<IterVar>;

/**
 * @brief Node for a reduce.
 * @author xupengcheng
 */
class Reduce : public Expr, public std::enable_shared_from_this<Reduce> {
 public:
  static constexpr IRNodeType type = IRNodeType::Reduce;
  Reduce(ExprPtr init, ExprPtr combiner, ScalarVarPtr accumulate,
         const ArrayPtr<IterVar> &reduce_axis, bool cross_thread = 0);

  ArrayPtr<IterVar> reduce_axis;
  ExprPtr init;
  ScalarVarPtr accumulate;
  ExprPtr combiner;
  bool cross_thread;
};
/**
 * @brief Pointer to Reduce object.
 * @author xupengcheng
 */
using ReducePtr = std::shared_ptr<Reduce>;

/**
 * @brief Node for a cast.
 * @author lixiuhong
 */
class Cast : public Expr, public std::enable_shared_from_this<Cast> {
 public:
  static constexpr IRNodeType type = IRNodeType::Cast;
  explicit Cast(ExprPtr need_cast_expr, ScalarType dst_type);
  ExprPtr expr_;  // the src Expr
};
/**
 * @brief Node for a cast.
 * @author lixiuhong
 */
using CastPtr = std::shared_ptr<Cast>;

enum SymbolType { Vecotr_Symbol, Broadcast_Symbol, Scalar_Symbol };

/**
 * @brief Node for a broadcast symbol.
 * @author mupei
 */
class BroadcastSymbol : public Expr,
                        public std::enable_shared_from_this<BroadcastSymbol> {
 public:
  static constexpr IRNodeType type = IRNodeType::BroadcastSymbol;
  explicit BroadcastSymbol(ExprPtr need_broadcast_expr,
                           std::stack<SymbolType> symbol_stack,
                           uint64_t lanes = 1);
  explicit BroadcastSymbol(ExprPtr need_broadcast_expr, uint64_t lanes = 1);

  uint64_t get_lanes() const;
  SymbolType pop_current_symbol();
  std::stack<SymbolType> get_current_symbol_stack() const;

  ExprPtr base_;  // the src Expr
 private:
  uint64_t lanes_;
  std::stack<SymbolType> symbol_stack_;
};
/**
 * @brief Node for a broadcast symbol.
 * @author mupei
 */
using BroadcastSymbolPtr = std::shared_ptr<BroadcastSymbol>;

/**
 * @brief Node for a vector symbol.
 * @author mupei
 */
class VectorSymbol : public Expr,
                     public std::enable_shared_from_this<VectorSymbol> {
 public:
  static constexpr IRNodeType type = IRNodeType::VectorSymbol;
  explicit VectorSymbol(ExprPtr base, std::stack<SymbolType> symbol_stack,
                        uint64_t stride = 1, uint64_t lanes = 1);

  uint64_t get_stride() const;
  uint64_t get_lanes() const;
  SymbolType pop_current_symbol();
  std::stack<SymbolType> get_current_symbol_stack() const;

  ExprPtr base_;

 private:
  uint64_t stride_;
  uint64_t lanes_;
  std::stack<SymbolType> symbol_stack_;
};
/**
 * @brief Node for a broadcast.
 * @author mupei
 */
using VectorSymbolPtr = std::shared_ptr<VectorSymbol>;

class Ramp : public Expr, public std::enable_shared_from_this<Ramp> {
 public:
  static constexpr IRNodeType type = IRNodeType::Ramp;
  explicit Ramp(ExprPtr base, int stride, int lanes);

  ExprPtr base;
  int stride;
  int lanes;
};

using RampPtr = std::shared_ptr<Ramp>;

// class BroadcastConst : public Expr,
//                        public std::enable_shared_from_this<BroadcastConst> {
//  public:
//   static constexpr IRNodeType type = IRNodeType::BroadcastConst;
//   explicit BroadcastConst(ExprPtr value, int lanes);

//   ExprPtr value;
//   int lanes;
// };

// using BroadcastConstPtr = std::shared_ptr<BroadcastConst>;

ExprPtr ceil_div(ExprPtr lhs, ExprPtr rhs);

}  // namespace ir

#endif  // ELENA_INCLUDE_IR_EXPR_H_
