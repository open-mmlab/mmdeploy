#include "IR/Expr.h"

#include "api.h"

namespace ir {

Expr::Expr(ScalarType dtype, IRNodeType type) : Node(type), dtype(dtype) {}

ScalarType Expr::get_dtype() const { return dtype; }

void Expr::set_dtype(ScalarType _dtype) { dtype = _dtype; }

Expr::~Expr() {}

ScalarAssign::ScalarAssign(ScalarVarPtr var, ExprPtr value):
    Expr(var->get_dtype(), type), var(var), value(value) {
    ELENA_ASSERT(var->get_dtype() == value->get_dtype(),
            "ScalarAssign Type Error.");
}

Binary::Binary(ExprPtr l, ExprPtr r, BinaryType binary_type)
    : Expr(l->get_dtype(), type), lhs(l), rhs(r), operation_type(binary_type) {
  // CHECK_SAME_DATA_TYPE(l, r)
  if (/*l->get_dtype() == ScalarType::Int8*/ true) {
    // as parrots sends Int8 as Boolean sometimes, this rule may be canceled for
    // bitwise op from parrots
    ELENA_ASSERT(true, "impossible to reach here");
  } else if (l->get_dtype() == ScalarType::Boolean) {
    ELENA_ASSERT(
        binary_type == BinaryType::And || binary_type == BinaryType::Or,
        "Only And and Or supported for Boolean Binary operands");
  } else {
    ELENA_ASSERT(
        binary_type != BinaryType::And && binary_type != BinaryType::Or,
        "And and Or not supported for numeric Binary operands");
  }
}

Unary::Unary(ExprPtr value, UnaryType unary_type)
    : Expr(value->get_dtype(), type),
      operand(value),
      operation_type(unary_type) {
  if (operand->get_dtype() == ScalarType::Boolean) {
    /*ELENA_ASSERT_EQ(unary_type, UnaryType::BitwiseNot,
              "Only Not supported for Boolean Unary operands");*/
  } else {
    /*ELENA_ASSERT_EQ(unary_type, UnaryType::Negate,
              "Only Negate supported for numeric Unary operands");*/
  }
}

Logical::Logical(ExprPtr l, ExprPtr r, LogicalType logical_type)
    : Expr(ScalarType::Boolean, type),
      lhs(l),
      rhs(r),
      operation_type(logical_type) {
  if (l->get_dtype() == ScalarType::Boolean) {
    ELENA_ASSERT(
        logical_type == LogicalType::AND || logical_type == LogicalType::OR
        || logical_type == LogicalType::EQ || logical_type == LogicalType::NE,
        "Only EQ and NE supported for Boolean Logical operands");
  }
}

Select::Select(ExprPtr cond, ExprPtr tBranch, ExprPtr fBranch)
    : Expr(tBranch->get_dtype(), type),
      cond(cond),
      tBranch(tBranch),
      fBranch(fBranch) {
  ELENA_ASSERT(cond->get_dtype() == ScalarType::Boolean
                        || cond->get_dtype() == ScalarType::UInt8,
                                       "Cast to Boolean before Select.");
  ELENA_ASSERT(tBranch->get_dtype() == fBranch->get_dtype(),
               "Incompatible branch types for 'tBranch' and 'fBranch' in "
               "'Select(cond, tBranch, fBranch)'.");
}

Call::Call(CallFunction func, ArrayPtr<Expr> args, ScalarType dtype)
    : Expr(dtype, type), func(func), args(args) {}

template <typename T>
Const<T>::Const(T value, ScalarType dtype) : Expr(dtype, type), value(value) {
  name = GENERATE_NAME(Const);
}

template <typename T>
Const<T>::~Const() {}

template <typename T>
T Const<T>::get_value() const {
  return value;
}

template <typename T>
std::string Const<T>::get_name() const {
  return name;
}

// explicit specialization
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, UNUSED) \
  template class Const<NATIVE_TYPE>;
#include "x/scalar_types.def"

Label::Label(const std::string &v) : Node(type), value(v) {}

const std::string &Label::get_value() const { return value; }

Var::Var(ScalarType dtype, IRNodeType type) : Expr(dtype, type) {}

Var::~Var() {}

NamedVar::NamedVar(std::string name, ScalarType dtype, IRNodeType type)
    : Var(dtype, type), var_name(name) {}

const std::string &NamedVar::get_name() const { return var_name; }

void NamedVar::set_name(const std::string &name) { var_name = name; }

ScalarVar::ScalarVar(ScalarType dtype)
    : ScalarVar(GENERATE_NAME(ScalarVar), dtype) {}

ScalarVar::ScalarVar(const std::string &name, ScalarType dtype)
    : NamedVar(name, dtype, type) {}

ScalarVar::ScalarVar(ExprPtr tensor, ArrayPtr<Expr> indices,
                     const std::string &name)
    : NamedVar(name, tensor->get_dtype(), type),
      tensor(ir::ptr_cast<TensorVar>(tensor)),
      indices(indices) {
  for (const auto &a : indices->element) {
    // CHECK_DATA_TYPE(a, UInt64)
    if (!(a->get_dtype() == ScalarType::UInt64
            || a->get_dtype() == ScalarType::Int64)) {
        ELENA_LOG_INFO("Type Error.");
    }
  }
}

ScalarVar::ScalarVar(ExprPtr tensor, ArrayPtr<Expr> indices)
    : NamedVar(GENERATE_NAME(ScalarVar), tensor->get_dtype(), type),
      tensor(ir::ptr_cast<TensorVar>(tensor)),
      indices(indices) {
  for (const auto &a : indices->element) {
    CHECK_DATA_TYPE(a, UInt64)
  }
}

bool ScalarVar::is_placeholder() const { return !tensor && !indices; }

TensorKey::TensorKey(std::string _key, ArrayPtr<Range> _bound) {
  key = _key;
  bound = _bound;
}

ArrayPtr<Range> TensorKey::get_bound() { return bound; }

std::string TensorKey::get_key() { return key; }

ExprPtr TensorVar::operator()(std::vector<ExprPtr> indices) {
  ELENA_ASSERT_EQ(indices.size(), shape->element.size(),
                  "Subscript indices should match tensor dimensions");
  return std::make_shared<ScalarVar>(shared_from_this(),
                                     std::make_shared<Array<Expr>>(indices),
                                     get_name() + "_slice");
}

Array<Expr> &TensorVar::get_shape() { return *shape; }

OpPtr TensorVar::get_op() { return op; }

Range::Range() : Node(type) {}

Range::Range(ExprPtr init, ExprPtr extent)
    : Node(type), init(init), extent(extent) {
  CHECK_DATA_TYPE(init, UInt64)
  CHECK_DATA_TYPE(extent, UInt64);
}

Range::Range(ExprPtr init, ExprPtr extent, ExprPtr stride)
    : Node(type), init(init), extent(extent), stride(stride) {
  CHECK_DATA_TYPE(init, UInt64);
  CHECK_DATA_TYPE(extent, UInt64);
  CHECK_DATA_TYPE(stride, UInt64);
}

Range::Range(const RangePtr &r) : Node(type), init(r->init), extent(r->extent) {
  if (r->stride)
    stride = r->stride;
  else
    stride = nullptr;
}

bool Range::is_null() { return init == nullptr; }

IterVar::IterVar(ExprPtr init, ExprPtr extent, const std::string &name,
                 bool is_reduce_)
    : IterVar(std::make_shared<Range>(init, extent), name, is_reduce_) {}

IterVar::IterVar(RangePtr range, const std::string &name, bool is_reduce_)
    : NamedVar(name, ScalarType::UInt64, type),
      range(range),
      is_reduce(is_reduce_) {}

Reduce::Reduce(ExprPtr init, ExprPtr combiner, ScalarVarPtr accumulate,
               const ArrayPtr<IterVar> &reduce_axis, bool cross_thread)
    : Expr(init->get_dtype(), type),
      reduce_axis(reduce_axis),
      init(init),
      accumulate(accumulate),
      combiner(combiner),
      cross_thread(cross_thread) {
  // dont check, some constant's type are ignorable in cuda
  // CHECK_SAME_DATA_TYPE(init, combiner)
  // CHECK_SAME_DATA_TYPE(combiner, accumulate)
  for (const auto &a : reduce_axis->element) {
    CHECK_DATA_TYPE(a, UInt64)
  }
}

Cast::Cast(ExprPtr need_cast_expr, ScalarType dst_type)
    : Expr(dst_type, type), expr_(need_cast_expr) {}

BroadcastSymbol::BroadcastSymbol(ExprPtr need_broadcast_expr,
                                 std::stack<SymbolType> symbol_stack,
                                 uint64_t lanes)
    : Expr(need_broadcast_expr->get_dtype(), type),
      base_(need_broadcast_expr),
      lanes_(lanes),
      symbol_stack_(symbol_stack) {}
uint64_t BroadcastSymbol::get_lanes() const { return lanes_; }

SymbolType BroadcastSymbol::pop_current_symbol() {
  SymbolType ret = symbol_stack_.top();
  symbol_stack_.pop();
  return ret;
}
std::stack<SymbolType> BroadcastSymbol::get_current_symbol_stack() const {
  return symbol_stack_;
}
BroadcastSymbol::BroadcastSymbol(ExprPtr need_broadcast_expr, uint64_t lanes)
    : Expr(need_broadcast_expr->get_dtype(), type),
      base_(need_broadcast_expr),
      lanes_(lanes) {}

VectorSymbol::VectorSymbol(ExprPtr base, std::stack<SymbolType> symbol_stack,
                           uint64_t stride, uint64_t lanes)
    : Expr(base->get_dtype(), type),
      base_(base),
      stride_(stride),
      lanes_(lanes),
      symbol_stack_(symbol_stack) {}
uint64_t VectorSymbol::get_lanes() const { return lanes_; }
uint64_t VectorSymbol::get_stride() const { return stride_; }
SymbolType VectorSymbol::pop_current_symbol() {
  SymbolType current_symbol = symbol_stack_.top();
  symbol_stack_.pop();
  return current_symbol;
}

Ramp::Ramp(ExprPtr base, int stride, int lanes)
    : Expr(base->get_dtype(), type), base(base), stride(stride), lanes(lanes) {}

// BroadcastConst::BroadcastConst(ExprPtr value, int lanes)
//     : Expr(value->get_dtype(), type), value(value), lanes(lanes) {}

std::stack<SymbolType> VectorSymbol::get_current_symbol_stack() const {
  return symbol_stack_;
}

TensorVar::TensorVar(std::shared_ptr<std::string> name, ArrayPtr<Expr> shape,
                     OpPtr op, ScalarType dtype)
    : Var(dtype, type), shape(shape), op(op), var_name(name) {}

TensorVar::TensorVar(ArrayPtr<Expr> shape, OpPtr op, ScalarType dtype)
    : TensorVar(std::make_shared<std::string>("tensor"), shape, op, dtype) {}

TensorVar::TensorVar(std::string &name, ArrayPtr<Expr> shape, OpPtr op,
                     ScalarType dtype)
    : TensorVar(std::shared_ptr<std::string>(op, &name), shape, op, dtype) {}

const std::string &TensorVar::get_name() const { return *var_name; }
void TensorVar::set_name(const std::string &name) { *var_name = name; }

ExprPtr operator+(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Add);
  return ExprPtr(op);
}

ExprPtr operator-(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Sub);
  return ExprPtr(op);
}

ExprPtr operator*(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Mul);
  return ExprPtr(op);
}

ExprPtr operator/(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Div);
  return ExprPtr(op);
}

ExprPtr operator%(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Mod);
  return ExprPtr(op);
}

ExprPtr operator-(ExprPtr operand) {
  Unary *op = new Unary(operand, UnaryType::Negate);
  return ExprPtr(op);
}

ExprPtr operator^(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Pow);
  return ExprPtr(op);
}

ExprPtr expr_and(ExprPtr lhs, ExprPtr rhs) {
  auto *op = new Logical(lhs, rhs, LogicalType::AND);
  return ExprPtr(op);
}

ExprPtr expr_or(ExprPtr lhs, ExprPtr rhs) {
  auto *op = new Logical(lhs, rhs, LogicalType::OR);
  return ExprPtr(op);
}

ExprPtr same(ExprPtr lhs, ExprPtr rhs) {
  auto *op = new Logical(lhs, rhs, LogicalType::EQ);
  return ExprPtr(op);
}

ExprPtr lt(ExprPtr lhs, ExprPtr rhs) {
  auto *op = new Logical(lhs, rhs, LogicalType::LT);
  return ExprPtr(op);
}

ExprPtr gt(ExprPtr lhs, ExprPtr rhs) {
  auto *op = new Logical(lhs, rhs, LogicalType::GT);
  return ExprPtr(op);
}

ExprPtr le(ExprPtr lhs, ExprPtr rhs) {
  auto *op = new Logical(lhs, rhs, LogicalType::LE);
  return ExprPtr(op);
}

ExprPtr ge(ExprPtr lhs, ExprPtr rhs) {
  auto *op = new Logical(lhs, rhs, LogicalType::GE);
  return ExprPtr(op);
}

ExprPtr and_(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::And);
  return ExprPtr(op);
}

ExprPtr or_(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Or);
  return ExprPtr(op);
}

ExprPtr max(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Max);
  return ExprPtr(op);
}

ExprPtr min(ExprPtr lhs, ExprPtr rhs) {
  Binary *op = new Binary(lhs, rhs, BinaryType::Min);
  return ExprPtr(op);
}

ExprPtr ceil_div(ExprPtr lhs, ExprPtr rhs) {
  ELENA_ASSERT_EQ(lhs->get_dtype(), rhs->get_dtype(),
                  "The scalartype should be the same.");
  ScalarType dtype = lhs->get_dtype();
  return std::make_shared<ir::Cast>(
      std::make_shared<ir::Unary>(
          std::make_shared<ir::Binary>(
              std::make_shared<ir::Cast>(lhs, ScalarType::Float32),
              std::make_shared<ir::Cast>(rhs, ScalarType::Float32),
              ir::BinaryType::Div),
          ir::UnaryType::Ceil),
      dtype);
}

#define TYPE_UNARYTYPE_FUNC_OP_MAP(BIG_ID, SMALL_ID) \
  ExprPtr SMALL_ID(ExprPtr lhs) {                    \
    Unary *op = new Unary(lhs, UnaryType::BIG_ID);   \
    return ExprPtr(op);                              \
  }
#include "x/unary_types.def"

}  // namespace ir
