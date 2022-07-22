#ifndef ELENA_INCLUDE_API_H_
#define ELENA_INCLUDE_API_H_

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "IR/Bound.h"
// #include "IR/Compute.h"
#include "IR/Container.h"
#include "IR/Expr.h"
#include "IR/Graph.h"
#include "IR/IRPrinter.h"
#include "IR/Node.h"
#include "IR/Op.h"
#include "IR/Stmt.h"
#include "IR/Type.h"
#include "Pass/Common/HoistIfThenElse.h"
#include "Pass/Common/InjectVirtualThread.h"
#include "Pass/Common/StatementSimplify.h"
#include "Pass/Common/StmtCopy.h"
#include "Pass/Common/StorageFlatten.h"
#include "Pass/Common/StorageRewrite.h"
#include "Pass/Common/Unroll.h"
#include "Pass/Common/VectorizeLoad.h"
#include "Pass/Hardware/SimdVectorize.h"
#include "Pass/Hardware/Tensorize.h"
#include "Schedule/Schedule.h"
#include "logging.h"

/**
 * @brief User-facing API for defining operators.
 * @author xupengcheng
 */
namespace api {

#define cast(x, type) std::make_shared<ir::Cast>(x, type)

using ir::Array;
using ir::Expr;
using ir::ExprPtr;
using ir::IterVar;
using ir::IterVarPtr;
using ir::MapPtr;
using ir::Range;
using ir::ScalarVarPtr;
using ir::SchedulePtr;
using ir::StmtPtr;
using ir::TensorVarPtr;
/**
 * @brief Create a scalar variable.  Template argument denotes the datatype
 * held by the variable.
 *
 * @author xupengcheng
 * @param name name for the variable.
 * @return the constructed ir::ScalarVar pointer.
 */
template <typename T>
ir::ExprPtr var(const std::string &name) = delete;

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  template <>                                                       \
  ir::ExprPtr var<NATIVE_TYPE>(const std::string &name);
#include "x/scalar_types.def"

template <typename T>
ir::ExprPtr var() = delete;

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  template <>                                                       \
  ir::ExprPtr var<NATIVE_TYPE>();
#include "x/scalar_types.def"

/**
 * @brief Create a tensor placeholder.
 *
 * Template argument denotes datatype held by the tensor.
 *
 * @author xupengcheng
 * @param shape shape for the tensor.
 * @param name name for the tensor.
 * @return the constructed ir::TensorVar pointer.
 */
template <typename T>
ir::TensorVarPtr placeholder(const ir::Array<ir::Expr> &shape,
                             const std::string &name) = delete;

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME)           \
  template <>                                                                 \
  ir::TensorVarPtr placeholder<NATIVE_TYPE>(const ir::Array<ir::Expr> &shape, \
                                            const std::string &name);
#include "x/scalar_types.def"

template <typename T>
ir::TensorVarPtr placeholder(const ir::Array<ir::Expr> &shape) = delete;

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  template <>                                                       \
  ir::TensorVarPtr placeholder<NATIVE_TYPE>(const ir::Array<ir::Expr> &shape);
#include "x/scalar_types.def"

/**
 * @brief Create a `numeric` Const expression node.
 *
 * The template is only specialized for supported constant types: the generic
 * template is deleted.
 *
 * @author xupengcheng
 * @param value value for the constant.
 * @return the constructed Const pointer.
 */
template <typename T>
ir::ExprPtr constant(T value) = delete;

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  template <>                                                       \
  ir::ExprPtr constant<NATIVE_TYPE>(NATIVE_TYPE value);
#include "x/scalar_types.def"

/**
 * @brief Test if given ir::Expr is a constant that equals the given constant
 * value.
 *
 * Only specializations for the currently supported constant types
 * provided.
 *
 * @author xupengcheng
 * @param expr the expression to be tested.
 * @param constant the desired constant.
 * @return true if the expression is a constant that equals the given constant;
 * false otherwise.
 */
template <typename T>
bool equal_const(ir::ExprPtr expr, T constant) = delete;

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  template <>                                                       \
  bool equal_const<NATIVE_TYPE>(ir::ExprPtr expr, NATIVE_TYPE constant);
#include "x/scalar_types.def"

/**
 * @brief Namespace for logical operations between expressions and logical
 * values that return Logical IR nodes.
 * @author xupengcheng
 */
namespace logical {
/**
 * @brief *Greater than or equal* node.
 * @author xupengcheng
 */
ir::ExprPtr ge(ir::ExprPtr lhs, ir::ExprPtr rhs);
/**
 * @brief *Greater than* node.
 * @author xupengcheng
 */
ir::ExprPtr gt(ir::ExprPtr lhs, ir::ExprPtr rhs);
/**
 * @brief *Less than or equal* node.
 * @author xupengcheng
 */
ir::ExprPtr le(ir::ExprPtr lhs, ir::ExprPtr rhs);
/**
 * @brief *Less than* node.
 * @author xupengcheng
 */
ir::ExprPtr lt(ir::ExprPtr lhs, ir::ExprPtr rhs);
/**
 * @brief *Equal* node.
 * @author xupengcheng
 */
ir::ExprPtr eq(ir::ExprPtr lhs, ir::ExprPtr rhs);
/**
 * @brief *Not equal* node.
 * @author xupengcheng
 */
ir::ExprPtr ne(ir::ExprPtr lhs, ir::ExprPtr rhs);

/**
 * @brief *Logical and* node.
 * @author xupengcheng
 */
ir::ExprPtr land(ir::ExprPtr lhs, ir::ExprPtr rhs);
/**
 * @brief *Logical or* node.
 * @author xupengcheng
 */
ir::ExprPtr lor(ir::ExprPtr lhs, ir::ExprPtr rhs);
/**
 * @brief *Logical not* node.
 * @author xupengcheng
 */
ir::ExprPtr lnot(ir::ExprPtr operand);

/**
 * @brief Logical and operation on a list of logical values.
 * @author xupengcheng
 * @param logicals list of all the logical values to be combined together with
 * logical and.
 * @return the constructed expression.
 */
ir::ExprPtr all(std::vector<ir::ExprPtr> logicals);

/**
 * @brief Logical or operation on a list of logical values.
 * @author xupengcheng
 * @param logicals list of all the logical values to be combined together with
 * logical or.
 * @return the constructed expression.
 */
ir::ExprPtr any(std::vector<ir::ExprPtr> logicals);
}  // namespace logical

/**
 * @brief Create iteration variables (indices) from output shape for
 * constructing fcompute for compute.
 *
 * The datatype for the output tensor follows fcompute.
 *
 * Example usage:
 *
 * ```
 * auto n = api::var("n");
 * auto iters = api::construct_indices({n});
 * ```
 *
 * @author xupengcheng
 * @param shape shape initializer_list for the desired output tensor.
 * @return vector containing the created iteration variables.
 */
ir::Array<ir::IterVar> construct_indices(const std::vector<ir::ExprPtr> &shape);

/**
 * @brief Create ir::IterVar for reduce axis.
 * @author xupengcheng
 * @param bounds lower and upper bound for reduce axis.
 * @param name name for the constructed ir::IterVar.
 * @return the constructed ir::IterVar.
 */
ir::IterVarPtr reduce_axis(std::pair<ir::ExprPtr, ir::ExprPtr> bounds,
                           const std::string &name);

/**
 * @brief Create Reduce for reduce expression.
 *
 * Refer to `api::sum` and other operations for example usage.
 *
 * `init`, `accumulate`, and `combiner` should have the same type.
 *
 * @author xupengcheng
 * @param init initial value for reduce
 * @param accumulate placeholder for accumulated value in combiner.
 * @param combiner combining function for reduce
 * @param reduce_axis list of reduce axes for reduce
 * @return the constructed Reduce.
 */
ir::ExprPtr reduce(ir::ExprPtr init, ir::ScalarVarPtr accumulate,
                   ir::ExprPtr combiner,
                   const ir::Array<ir::IterVar> &reduce_axis,
                   bool cross_thread = 0);

/**
 * @brief The thread_level reduce operation.
 *
 * Only support four protocol reductions: sum, max, min
 *
 * @author zhuqianchao
 * @param expr expression to multiply.
 * @param reduce_axis reduction axes to follow.
 * @param reduce_op type of reduction operation.
 * @return expression for the product result.
 */
template <typename T>
ir::ExprPtr cross_thread_reduce(ir::ExprPtr expr,
                                const ir::Array<ir::IterVar> &reduce_axis,
                                const std::string &reduce_op);

/**
 * @brief The sum reduce operation.
 *
 * Special case of reduce with init=0, combiner=`+`.  The template argument
 * denotes type of the constant for init used.
 *
 * @author xupengcheng
 * @param expr expression to sum.
 * @param reduce_axis reduction axes to follow.
 * @return expression for the sum result.
 */
template <typename T>
ir::ExprPtr sum(ir::ExprPtr expr, const ir::Array<ir::IterVar> &reduce_axis);

/**
 * @brief The max reduce operation.
 *
 * Special case of reduce with init=mini, combiner=`max`.  The template argument
 * denotes type of the constant for init used.
 *
 * @author xiaoquanlun
 * @param expr expression to max.
 * @param reduce_axis reduction axes to follow.
 * @return expression for the ,ax result.
 */
template <typename T>
ir::ExprPtr max_reduce(ir::ExprPtr expr,
                       const ir::Array<ir::IterVar> &reduce_axis);

/**
 * @brief The min reduce operation.
 *
 * Special case of reduce with init=maxi, combiner=`min`.  The template argument
 * denotes type of the constant for init used.
 *
 * @author xiaoquanlun
 * @param expr expression to min.
 * @param reduce_axis reduction axes to follow.
 * @return expression for the ,ax result.
 */
template <typename T>
ir::ExprPtr min_reduce(ir::ExprPtr expr,
                       const ir::Array<ir::IterVar> &reduce_axis);

/**
 * @brief The product reduce operation.
 *
 * Special case of reduce with init=1, combiner=`*`.  The template argument
 * denotes type of the constant for init used.
 *
 * @author xupengcheng
 * @param expr expression to multiply.
 * @param reduce_axis reduction axes to follow.
 * @return expression for the product result.
 */
template <typename T>
ir::ExprPtr product(ir::ExprPtr expr,
                    const ir::Array<ir::IterVar> &reduce_axis);

/**
 * @brief Call function operation include call_intrin and call_function
 *
 * Only support function in call_function_types.def
 *
 * call_intrin/call_function
 * @author zhuqianchao
 * @param call_function name
 * @param attached Tensorvar(Called after tensor calculation by default) for
 * call_intrin
 * @param arguments for call function
 * @return void / ir::Call ExprPtr with call function
 */

void call_intrin(const std::string &function_name,
                 const ir::TensorVarPtr &attached_var);

template <typename...>
struct accepts : std::false_type {};
template <>
struct accepts<ir::ExprPtr, ir::ExprPtr> : std::true_type {};
// template<> struct accepts<ir::ExprPtr>: std::true_type {};

template <typename... T>
ExprPtr call_function(const std::string &function_name, T... args) {
  static_assert(
      sizeof...(args) > 0 || accepts<T...>::value,
      "At least one argument is required or the argument is not accepted");
  ExprPtr intrin_expr;
  if (function_name == "atomicAdd") {
    ELENA_ASSERT_EQ(sizeof...(args), 2, "Atomic opreation need 2 arguments");
    std::vector<ir::ExprPtr> atomic_args{args...};
    intrin_expr = std::make_shared<ir::Call>(
        ir::CallFunction::atomic_add,
        std::make_shared<ir::Array<ir::Expr>>(atomic_args),
        atomic_args[0]->get_dtype());
  } else if (function_name == "atomicMax") {
    ELENA_ASSERT_EQ(sizeof...(args), 2, "Atomic opreation need 2 arguments");
    std::vector<ir::ExprPtr> atomic_args{args...};
    intrin_expr = std::make_shared<ir::Call>(
        ir::CallFunction::atomic_max,
        std::make_shared<ir::Array<ir::Expr>>(atomic_args),
        atomic_args[0]->get_dtype());
  } else if (function_name == "atomicMin") {
    ELENA_ASSERT_EQ(sizeof...(args), 2, "Atomic opreation need 2 arguments");
    std::vector<ir::ExprPtr> atomic_args{args...};
    intrin_expr = std::make_shared<ir::Call>(
        ir::CallFunction::atomic_min,
        std::make_shared<ir::Array<ir::Expr>>(atomic_args),
        atomic_args[0]->get_dtype());
  } else {
    ELENA_ABORT(
        "The currently supported intrinsic include {atomicAdd, atomicMax, "
        "atomicMin}");
  }
  return intrin_expr;
};

template <typename... T>
void call_intrin(const std::string &function_name,
                 const ir::TensorVarPtr &attached_var, T... args) {
  static_assert(
      sizeof...(args) > 0 || accepts<T...>::value,
      "At least one argument is required or the argument is not accepted");
  ExprPtr intrin_expr = call_function(function_name, args...);
  attached_var->op->attached_intrin_expr = intrin_expr;
};

/**
 * @brief Create a ComputeOp and return its output ir::TensorVar.
 *
 * The iteration variables used by the fcompute expression should be created
 * first via `api::construct_indices`.  Example usage:
 *
 * ```
 * auto A = api::placeholder({n}, "A");
 * auto iters = api::construct_indices({n});
 * auto out = api::compute({n}, iters, (*A));
 * ```
 *
 * @author xupengcheng
 * @param shape shape initializer_list for the desired output tensor.
 * @param iter_vars iteration variables for constructing the fcompute
 * expression.
 * @param fcompute compute expression for each element in the output tensor.
 * The expression is a function of the iteration variables specified in
 * `iter_vars`.
 * @param name name of the output ir::TensorVar.
 * @return output tensor for the created ComputeOp.
 */
ir::TensorVarPtr compute(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> iter_vars, ir::ExprPtr fcompute,
                         const std::string &name);

ir::TensorVarPtr compute(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> iter_vars,
                         ir::ExprPtr fcompute);

ir::TensorVarPtr compute(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> for_iter_vars,
                         ir::Array<ir::Expr> tensor_indices,
                         ir::ExprPtr fcompute, const std::string &name);

/**
 * @brief If-then-else (select) operation.
 *
 * If the condition is true, the expression's value is taken from the first
 * argument; otherwise the expression's value is taken from the second argument.
 *
 * @author xupengcheng
 * @param condition condition expression.
 * @param then_value value when the condition is true.
 * @param else_value value when the condition is false.
 * @return the constructed expression.
 */
ir::ExprPtr if_then_else(ir::ExprPtr condition, ir::ExprPtr then_value,
                         ir::ExprPtr else_value);

/**
 * @brief Print the Node.
 * @author hanruobing
 * @param node node need to print
 * @param os `ostream` to write output to.
 * @param verbose whether show details of a node.
 */
void dump_ast(ir::NodePtr node, std::ostream &os, bool verbose = false);

// TODO(hanruobing) :write dump_stmt for different IR in a concise method
/**
 * @brief Print the Stmt.
 * @author hanruobing
 * @param stmt Stmt need to print
 * @param os `ostream` to write output to.
 * @param verbose whether show details of a node.
 */
void dump_stmt(StmtPtr stmt, std::ostream &os, bool verbose = false);

/**
 * @brief Dump expr.
 * @author hanruobing
 * @param expr Expr need to print
 * @param os 'ostream' to write output to
 * @param verbose whether show details of a node.
 */
void dump_expr(ExprPtr expr, std::ostream &os, bool verbose = false);

/**
 * @brief Create a new schedule for given Ops
 * @author lichuandong
 * @param ops op ir::Array for construct the schedule
 * @return the new schedule
 */
ir::SchedulePtr create_schedule(ir::Array<ir::Op> ops);

/**
 * @brief Create a new schedule for given Op
 * @author lichuandong
 * @param op op for construct the schedule
 * @return the new schedule
 */
ir::SchedulePtr create_schedule(ir::OpPtr op);

/**
 * @brief Print details from the given schedule
 * @author lichuandong
 * @param schedule the schedule to print
 */
void print_schedule(ir::SchedulePtr schedule, std::ostream &os);

/**
 * @brief Infer iteration's bound
 * @author hanruobing
 * @param schedule the schedule needs to infer
 */
MapPtr<IterVar, Range> inferBound(ir::SchedulePtr schedule);

/**
 * @brief StorageFlatten operation
 * @author xiaoquanlun
 * @param stmt the root node ptr to be flattened
 * @param bound output from inferBound
 */

ir::StmtPtr flattenStorage(ir::StmtPtr stmt, MapPtr<IterVar, Range> bound);

/**
 * @brief StorageRewrite operation
 * @author hanruobing
 * @param stmt the root node ptr to be rewritten
 */

ir::StmtPtr rewriteStorage(ir::StmtPtr stmt);

/**
 * @brief LoopVectorizer operation
 * @author mupei
 * @param stmt the root node ptr to be vectorize
 */
ir::StmtPtr simdVectorizeLoop(ir::StmtPtr stmt,
                              const bool enable_vectorize = true);

/**
 * @brief Transforms a schedule to stmt
 * @author hanruobing
 * @param sch Schedule that to be transformed
 * @return correspond stmt
 */
ir::StmtPtr scheduleToStatement(ir::SchedulePtr sch,
                                ir::MapPtr<ir::IterVar, ir::Range> dom_map);

/**
 * @brief Simplify an expression.
 * @author guanzhichao
 * @param expr expression to simplify.
 * @return the simplified expression.
 */
ir::ExprPtr simplify(ir::ExprPtr expr);
ir::StmtPtr simplify(ir::StmtPtr node);

/**
 * @brief generate cuda header(fixed, not dependent on elena ir for now) for
 * parrots
 * @author xiaoquanlun
 * @return the header string on cuda
 */
std::string genCudaHeader(int float_type = 0,
                          const std::string &hname = "elena_int");

/**
 * @brief CUDA-ify an IR tree.
 * The in/out variables would be named as Var<id>.
 * @author xieruifeng
 * @param node The root node of the IR tree.
 * @param arg_list      The input/output placeholders, of type [<id, type>].
 * @param kernel_name   The name of the generated function (aka kernel).
 * @return std::string The generated CUDA code.
 *
 * @warning Assumes all variables is named as Var<id>.
 * @warning Do not check whether the outmost allocate match the output name.
 */
std::string genCudaSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name);  // delete this implementation next

std::string genCudaSrc(const ir::NodePtr &node,
                       const std::vector<ir::ExprPtr> &arg_list,
                       const std::string &kernel_name,
                       const std::vector<ir::ScalarVarPtr> &varlist = {});

/**
 * @brief Generate x86 source code for the IR.
 * The in/out variables would be named as Var<id>.
 * @author xieruifeng
 * @param node The root node of the IR tree.
 * @param arg_list      The input/output placeholders, of type [<id, type>].
 * @param kernel_name   The name of the generated function (aka kernel).
 * @return std::string The generated CUDA code.
 *
 * @warning Assumes all variables is named as Var<id>.
 * @warning Do not check whether the outmost allocate match the output name.
 */
std::string genX86Src(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name);



std::string genX86Src(const ir::NodePtr &node,
                       const std::vector<ir::ExprPtr> &arg_list,
                       const std::string &kernel_name,
                       const std::vector<ir::ScalarVarPtr> &varlist = {});


/**
 * @brief Generate raw x86 code without a schedule.
 * @param var The result `TensorVar`.
 * @param args The input 'TensorVar's, of type [TensorVarPtr], `var` included.
 * @return the generated x86 code as `std::string`.
 */
std::string genX86Raw(const ir::TensorVarPtr &var,
                      const std::vector<TensorVarPtr> &args);

/**
 * @brief generate hip header(fixed, not dependent on elena ir for now) for
 * parrots
 * @author hubinbin
 * @return the header string on hip
 */
std::string genHipHeader(std::string hname = "elena_int");

/**
 * @brief HIP-ify an IR tree.
 * The in/out variables would be named as Var<id>.
 * @author hubinbin
 * @param node The root node of the IR tree.
 * @param arg_list      The input/output placeholders, of type [<id, type>].
 * @param kernel_name   The name of the generated function (aka kernel).
 * @param ostr The generated HIP code would be written to this stream.
 *
 * @warning Assumes all variables is named as Var<id>.
 * @warning Do not check whether the outmost allocate match the output name.
 */
void genHipSrc(const ir::NodePtr &node,
               const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
               std::string kernel_name, std::ostream &ostr);
/**
 * @brief HIP-ify an IR tree.
 * The in/out variables would be named as Var<id>.
 * @author hubinbin
 * @param node The root node of the IR tree.
 * @param arg_list      The input/output placeholders, of type [<id, type>].
 * @param kernel_name   The name of the generated function (aka kernel).
 * @return std::string The generated HIP code.
 *
 * @warning Assumes all variables is named as Var<id>.
 * @warning Do not check whether the outmost allocate match the output name.
 */
std::string genHipSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name);

void dump_code(std::string code, std::string file_name = "source.cu");

void dump_diminfo(int bx, int by, int bz, int tx, int ty, int tz);

std::string genHostSrc(std::vector<std::pair<int, ir::ScalarType>> arg_list,
                       std::vector<int64_t> tensor_size,
                       std::string kernel_name, std::vector<int> dim_info,
                       backend::TargetType target);

/**
 * @brief generate BANG header for BANG
 * @author mupei
 * @return the header string on Cambricon
 */
std::string genBangHeader(std::string hname = "elena_int");

/**
 * @brief BANTG-ify an IR tree.
 * The in/out variables would be named as Var<id>.
 * @author mupei
 * @param node The root node of the IR tree.
 * @param arg_list      The input/output placeholders, of type [<id, type>].
 * @param kernel_name   The name of the generated function (aka kernel).
 * @param ostr The generated BANG code would be written to this stream.
 *
 * @warning Assumes all variables is named as Var<id>.
 * @warning Do not check whether the outmost allocate match the output name.
 */
void genBangSrc(const ir::NodePtr &node,
                const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
                std::string kernel_name, std::ostream &ostr);
/**
 * @brief BANTG-ify an IR tree.
 * The in/out variables would be named as Var<id>.
 * @author mupei
 * @param node The root node of the IR tree.
 * @param arg_list      The input/output placeholders, of type [<id, type>].
 * @param kernel_name   The name of the generated function (aka kernel).
 * @return std::string The generated BANG code.
 *
 * @warning Assumes all variables is named as Var<id>.
 * @warning Do not check whether the outmost allocate match the output name.
 */
std::string genBangSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name);

std::string genTangSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name);

std::string genTangSrc(
    const ir::NodePtr &node,
    const std::vector<ir::ExprPtr> &arg_list,
    const std::string &kernel_name);

std::string genTangHeader();

}  // namespace api

#endif  // ELENA_INCLUDE_API_H_
