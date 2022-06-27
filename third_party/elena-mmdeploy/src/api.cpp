#include "api.h"


#include <fstream>

#include "IR/NameGenerator.h"

namespace api {

using ir::AttachType;
using ir::BinaryType;
using ir::CallFunction;
using ir::IRNodeType;
using ir::LogicalType;
using ir::Node;
using ir::ScalarType;
using ir::UnaryType;

#define IR_NODE_TYPE(IRNODETYPE) \
  using ir::IRNODETYPE;          \
  using ir::IRNODETYPE##Ptr;
#include "x/ir_node_types.def"

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME)          \
  template <>                                                                \
  ir::ExprPtr var<NATIVE_TYPE>(const std::string &name) {                    \
    return std::make_shared<ir::ScalarVar>(name,                             \
                                           ir::ScalarType::SCALARTYPE_NAME); \
  }
#include "x/scalar_types.def"

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME)          \
  template <>                                                                \
  ir::ExprPtr var<NATIVE_TYPE>() {                                           \
    return std::make_shared<ir::ScalarVar>(ir::ScalarType::SCALARTYPE_NAME); \
  }
#include "x/scalar_types.def"

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME)            \
  template <>                                                                  \
  ir::TensorVarPtr placeholder<NATIVE_TYPE>(const ir::Array<ir::Expr> &shape,  \
                                            const std::string &name) {         \
    return PlaceholderOp::create(std::make_shared<ir::Array<ir::Expr>>(shape), \
                                 ir::ScalarType::SCALARTYPE_NAME, name)        \
        ->output(0);                                                           \
  }
#include "x/scalar_types.def"

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME)            \
  template <>                                                                  \
  ir::TensorVarPtr placeholder<NATIVE_TYPE>(                                   \
      const ir::Array<ir::Expr> &shape) {                                      \
    return PlaceholderOp::create(std::make_shared<ir::Array<ir::Expr>>(shape), \
                                 ir::ScalarType::SCALARTYPE_NAME)              \
        ->output(0);                                                           \
  }
#include "x/scalar_types.def"

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  template <>                                                       \
  ir::ExprPtr constant<NATIVE_TYPE>(NATIVE_TYPE value) {            \
    return std::make_shared<Const<NATIVE_TYPE>>(                    \
        value, ir::ScalarType::SCALARTYPE_NAME);                    \
  }
#include "x/scalar_types.def"

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME)       \
  template <>                                                             \
  bool equal_const<NATIVE_TYPE>(ir::ExprPtr expr, NATIVE_TYPE constant) { \
    if (expr->get_type() != ir::IRNodeType::Const) {                      \
      return false;                                                       \
    }                                                                     \
    if (expr->get_dtype() != ir::ScalarType::SCALARTYPE_NAME) {           \
      return false;                                                       \
    }                                                                     \
    auto const_ptr = ir::ptr_cast<Const<NATIVE_TYPE>>(expr);              \
    return constant == const_ptr->get_value();                            \
  }
#include "x/scalar_types.def"

namespace logical {

ir::ExprPtr ge(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Logical>(lhs, rhs, ir::LogicalType::GE);
}

ir::ExprPtr gt(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Logical>(lhs, rhs, ir::LogicalType::GT);
}

ir::ExprPtr le(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Logical>(lhs, rhs, ir::LogicalType::LE);
}

ir::ExprPtr lt(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Logical>(lhs, rhs, ir::LogicalType::LT);
}

ir::ExprPtr eq(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Logical>(lhs, rhs, ir::LogicalType::EQ);
}

ir::ExprPtr ne(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Logical>(lhs, rhs, ir::LogicalType::NE);
}

ir::ExprPtr land(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Binary>(lhs, rhs, ir::BinaryType::And);
}

ir::ExprPtr lor(ir::ExprPtr lhs, ir::ExprPtr rhs) {
  return std::make_shared<ir::Binary>(lhs, rhs, ir::BinaryType::Or);
}

ir::ExprPtr lnot(ir::ExprPtr operand) {
  return std::make_shared<ir::Unary>(operand, ir::UnaryType::BitwiseNot);
}

ir::ExprPtr all(std::vector<ir::ExprPtr> logicals) {
  ir::ExprPtr acc = api::constant<bool>(true);
  for (const auto &a : logicals) {
    acc = land(acc, a);
  }
  return acc;
}

ir::ExprPtr any(std::vector<ir::ExprPtr> logicals) {
  ir::ExprPtr acc = api::constant<bool>(false);
  for (const auto &a : logicals) {
    acc = lor(acc, a);
  }
  return acc;
}

}  // namespace logical

ir::Array<ir::IterVar> construct_indices(
    const std::vector<ir::ExprPtr> &shape) {
  ir::Array<ir::IterVar> iter_vars;
  ir::ExprPtr init = api::constant<uint64_t>(0);
  static int i = 0;
  for (const auto &a : shape) {
    if (a->get_type() == ir::IRNodeType::IterVar) {
      iter_vars.element.emplace_back(ir::ptr_cast<ir::IterVar>(a));
    } else {
      CHECK_DATA_TYPE(a, UInt64)
      ir::ExprPtr extent = a;
      iter_vars.element.emplace_back(new ir::IterVar(
          init, extent, std::string("iter") + std::to_string(i++)));
    }
  }
  return iter_vars;
}

ir::IterVarPtr reduce_axis(std::pair<ir::ExprPtr, ir::ExprPtr> bounds,
                           const std::string &name) {
  const auto &init = bounds.first;
  const auto &extent = bounds.second;
  return std::make_shared<ir::IterVar>(init, extent, name, true);
}

ir::ExprPtr reduce(ir::ExprPtr init, ir::ScalarVarPtr accumulate,
                   ir::ExprPtr combiner,
                   const ir::Array<ir::IterVar> &reduce_axis,
                   bool cross_thread) {
  return std::make_shared<ir::Reduce>(
      init, combiner, accumulate,
      std::make_shared<ir::Array<ir::IterVar>>(reduce_axis), cross_thread);
}

template <typename T>
ir::ExprPtr cross_thread_reduce(ir::ExprPtr expr,
                                const ir::Array<ir::IterVar> &reduce_axis,
                                const std::string &reduce_op) {
  bool cross_thread = true;
  if (reduce_op == "sum") {
    auto init = api::constant<T>(0);
    auto accumulate =
        std::make_shared<ir::ScalarVar>("sum_accumulate", expr->get_dtype());
    auto combiner = accumulate + expr;
    return api::reduce(init, accumulate, combiner, reduce_axis, cross_thread);
  } else if (reduce_op == "max") {
    ExprPtr init;
    if (std::is_same<T, char>::value) {
      init =
          api::constant<T>(static_cast<int>(std::numeric_limits<T>::lowest()));
    } else {
      init = api::constant<T>(std::numeric_limits<T>::lowest());
    }
    auto accumulate =
        std::make_shared<ir::ScalarVar>("max_accumulate", expr->get_dtype());
    auto combiner = max(accumulate, expr);
    return api::reduce(init, accumulate, combiner, reduce_axis, cross_thread);
  } else if (reduce_op == "min") {
    ExprPtr init;
    if (std::is_same<T, char>::value) {
      init = api::constant<T>(static_cast<int>(std::numeric_limits<T>::max()));
    } else {
      init = api::constant<T>(std::numeric_limits<T>::max());
    }
    auto accumulate =
        std::make_shared<ir::ScalarVar>("min_accumulate", expr->get_dtype());
    auto combiner = min(accumulate, expr);
    return api::reduce(init, accumulate, combiner, reduce_axis, cross_thread);
  } else {
    ELENA_ABORT("The currently supported reduction type: sum, max, min");
  }
}

template <typename T>
ir::ExprPtr sum(ir::ExprPtr expr, const ir::Array<ir::IterVar> &reduce_axis) {
  auto init = api::constant<T>(0);
  auto accumulate =
      std::make_shared<ir::ScalarVar>("sum_accumulate", expr->get_dtype());
  auto combiner = accumulate + expr;
  return api::reduce(init, accumulate, combiner, reduce_axis);
}

template <typename T>
ir::ExprPtr max_reduce(ir::ExprPtr expr,
                       const ir::Array<ir::IterVar> &reduce_axis) {
  ExprPtr init;
  if (std::is_same<T, char>::value) {
    init = api::constant<T>(static_cast<int>(std::numeric_limits<T>::lowest()));
  } else {
    init = api::constant<T>(std::numeric_limits<T>::lowest());
  }
  auto accumulate =
      std::make_shared<ir::ScalarVar>("max_accumulate", expr->get_dtype());
  auto combiner = max(accumulate, expr);
  return api::reduce(init, accumulate, combiner, reduce_axis);
}

template <typename T>
ir::ExprPtr min_reduce(ir::ExprPtr expr,
                       const ir::Array<ir::IterVar> &reduce_axis) {
  ExprPtr init;
  if (std::is_same<T, char>::value) {
    init = api::constant<T>(static_cast<int>(std::numeric_limits<T>::max()));
  } else {
    init = api::constant<T>(std::numeric_limits<T>::max());
  }
  auto accumulate =
      std::make_shared<ir::ScalarVar>("min_accumulate", expr->get_dtype());
  auto combiner = min(accumulate, expr);
  return api::reduce(init, accumulate, combiner, reduce_axis);
}

template <typename T>
ir::ExprPtr product(ir::ExprPtr expr,
                    const ir::Array<ir::IterVar> &reduce_axis) {
  auto init = api::constant<T>(1);
  auto accumulate =
      std::make_shared<ir::ScalarVar>("product_accumulate", expr->get_dtype());
  auto combiner = accumulate * expr;
  return api::reduce(init, accumulate, combiner, reduce_axis);
}

// explicit instantiation
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, UNUSED)          \
  template ir::ExprPtr cross_thread_reduce<NATIVE_TYPE>(            \
      ir::ExprPtr expr, const ir::Array<ir::IterVar> &reduce_axis,  \
      const std::string &reduce_op);                                \
  template ir::ExprPtr sum<NATIVE_TYPE>(                            \
      ir::ExprPtr expr, const ir::Array<ir::IterVar> &reduce_axis); \
  template ir::ExprPtr max_reduce<NATIVE_TYPE>(                     \
      ir::ExprPtr expr, const ir::Array<ir::IterVar> &reduce_axis); \
  template ir::ExprPtr min_reduce<NATIVE_TYPE>(                     \
      ir::ExprPtr expr, const ir::Array<ir::IterVar> &reduce_axis); \
  template ir::ExprPtr product<NATIVE_TYPE>(                        \
      ir::ExprPtr expr, const ir::Array<ir::IterVar> &reduce_axis);
#include "x/scalar_types.def"

void call_intrin(const std::string &function_name,
                 const ir::TensorVarPtr &attached_var) {
  ExprPtr intrin_expr;
  if (function_name == "sync") {
    intrin_expr = std::make_shared<ir::Call>(ir::CallFunction::Sync, nullptr,
                                             ir::ScalarType::Boolean);
  } else {
    ELENA_ABORT("not support this intrin");
  }
  attached_var->op->attached_intrin_expr = intrin_expr;
}

ir::TensorVarPtr compute(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> iter_vars, ir::ExprPtr fcompute,
                         const std::string &name) {
  ir::OpPtr op = ir::ComputeOp::create(
      std::make_shared<ir::Array<ir::Expr>>(shape),
      std::make_shared<ir::Array<ir::IterVar>>(iter_vars), fcompute, name);
  return op->output(0);
}

ir::TensorVarPtr compute(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> iter_vars,
                         ir::ExprPtr fcompute) {
  ir::OpPtr op = ir::ComputeOp::create(
      std::make_shared<ir::Array<ir::Expr>>(shape),
      std::make_shared<ir::Array<ir::IterVar>>(iter_vars), fcompute);
  return op->output(0);
}

ir::TensorVarPtr compute(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> for_iter_vars,
                         ir::Array<ir::Expr> tensor_indices,
                         ir::ExprPtr fcompute, const std::string &name) {
  if (shape.size() != tensor_indices.size())
    ELENA_ABORT("tensor indices must have same dimension with shape");

  ir::Array<ir::IterVar> tensor_itervar;
  static int i = 0;
  for (auto &iter : tensor_indices.element) {
    if (iter->get_type() == IRNodeType::IterVar) {
      auto iter_var = ir::ptr_cast<IterVar>(iter);
      tensor_itervar.element.emplace_back(iter_var);
    } else {
      ir::ExprPtr const_iter = iter;
      tensor_itervar.element.emplace_back(
          new ir::IterVar(const_iter, const_iter,
                          std::string("const_iter") + std::to_string(i++)));
    }
  }

  ir::OpPtr op = ir::ComputeOp::create(
      std::make_shared<ir::Array<ir::Expr>>(shape),
      std::make_shared<ir::Array<ir::IterVar>>(for_iter_vars),
      std::make_shared<ir::Array<ir::IterVar>>(tensor_itervar), fcompute, name);
  return op->output(0);
}

ir::ExprPtr if_then_else(ir::ExprPtr condition, ir::ExprPtr then_value,
                         ir::ExprPtr else_value) {
  CHECK_DATA_TYPE(condition, Boolean)
  CHECK_SAME_DATA_TYPE(then_value, else_value)
  return std::make_shared<ir::Select>(condition, then_value, else_value);
}

void dump_ast(ir::NodePtr node, std::ostream &os, bool verbose) {
  if (verbose) {
    IRPrinter printer(os);
    printer.print(node.get());
  } else {
    SimpleIRPrinter printer(os);
    printer.print(node.get());
  }
  os << std::endl;
}

void dump_expr(ExprPtr expr, std::ostream &os, bool verbose) {
  dump_ast(expr, os, verbose);
}

void dump_stmt(StmtPtr stmt, std::ostream &os, bool verbose) {
  dump_ast(stmt, os, verbose);
}

ir::SchedulePtr create_schedule(ir::Array<ir::Op> ops) {
  return std::make_shared<Schedule>(std::make_shared<ir::Array<ir::Op>>(ops));
}

ir::SchedulePtr create_schedule(ir::OpPtr op) {
  return std::make_shared<Schedule>(
      std::make_shared<ir::Array<ir::Op>>(std::initializer_list<OpPtr>{op}));
}

void print_schedule(ir::SchedulePtr schedule, std::ostream &os) {
  IRPrinter printer(os);
  printer.print(schedule.get());
  os << std::endl;
}

MapPtr<IterVar, Range> inferBound(ir::SchedulePtr schedule) {
  graph::ReadGraph read_graph = graph::CreateReadGraph(schedule->outputs);
  graph::FeedGraph feed_graph = graph::CreateFeedGraph(read_graph);

  auto attach_path = graph::CreateAttachPath(schedule);
  std::unordered_map<ir::OpPtr, ir::StagePtr> op2stage;
  for (auto stage : schedule->stages->element) {
    op2stage[stage->op] = stage;
  }

  // Run inference.
  MapPtr<IterVar, Range> ret = std::make_shared<ir::Map<IterVar, Range>>();
  for (size_t i = schedule->stages->size(); i != 0; --i) {
    const StagePtr stage = schedule->stages->element[i - 1];

    inferRootBound(stage, feed_graph, op2stage, attach_path, ret);

    // pass down to get bound of all iter vars.
    passDownDomain(stage, ret);
  }
  return ret;
}

ir::StmtPtr flattenStorage(ir::StmtPtr stmt, MapPtr<IterVar, Range> bound) {
  auto flattenStorageer = std::make_shared<StorageFlattener>(bound);
  auto flattened_node = flattenStorageer->mutateReplace(stmt);
  auto flattened_stmt = ir::ptr_cast<ir::Stmt>(flattened_node);
  return flattened_stmt;
}

ir::StmtPtr rewriteStorage(ir::StmtPtr stmt) {
  StoragePlanRewriter writer;
  return writer.rewrite(stmt);
}

ir::StmtPtr simdVectorizeLoop(ir::StmtPtr stmt, const bool enable_vectorize) {
  ELENA_ASSERT(stmt != nullptr, "Stmt is not exit!");
  if (enable_vectorize) {
    ir::LoopVectorizer().mutate(stmt);
  } else {
    ir::LoopVectorizerSkipper().mutate(stmt);
  }
  return stmt;
}

// inject the operator's realization on the stmt.
class InjectAttach : public MutatorBase<InjectAttach> {
 public:
  InjectAttach(const StagePtr stage_, const IterVarPtr attached_iter_,
               MapPtr<IterVar, Range> dom_map_)
      : stage(stage_), attached_iter(attached_iter_), dom_map(dom_map_) {}

  ir::NodePtr visit(ir::For *node) {
    auto stmt = MutatorBase::visit(node);
    auto for_ptr = ir::ptr_cast<For>(stmt);

    if (for_ptr->it == attached_iter) {
      find_attach = true;
      stmt = makePipeline(stage, dom_map, ir::ptr_cast<Stmt>(for_ptr->body));
      for_ptr->body = ir::ptr_cast<Stmt>(stmt);
    }
    return for_ptr;
  }
  ir::NodePtr visit(ir::Attr *node) {
    auto stmt = MutatorBase::visit(node);
    auto attr_ptr = ir::ptr_cast<Attr>(stmt);

    if (attr_ptr->node == attached_iter) {
      find_attach = true;
      stmt = makePipeline(stage, dom_map, ir::ptr_cast<Stmt>(attr_ptr->body));
      attr_ptr->body = ir::ptr_cast<Stmt>(stmt);
    }
    return attr_ptr;
  }
  using MutatorBase::visit;

  StmtPtr inject_attach(StmtPtr body) {
    find_attach = false;
    mutate(body);
    if (!find_attach) {
      std::stringstream sstr;
      sstr << "error: do not find attach point for "
           << stage->op->output(0)->get_name() << std::endl;
      std::string str;
      sstr >> str;
      ELENA_WARN(str.c_str());
      abort();
    }
    return body;
  }

 private:
  // The stage.
  const StagePtr stage;
  // The attach spec, may not contain op.
  const IterVarPtr attached_iter;
  // domain map
  const MapPtr<IterVar, Range> dom_map;
  // flag for find compute node
  bool find_attach = false;
};

// delete duplicated thread extent attr
class SchedulePostProc : public MutatorBase<SchedulePostProc> {
 public:
  /// Visit instances of class Attr and delete duplicated thread
  /// extent attr.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(attr_ptr);
  /// \encode
  ///
  /// \param attr_ptr pointer to the instance of class Attr;
  ///
  /// \return the body of the instance of class Attr.
  ir::NodePtr visit(ir::Attr *attr_ptr) {
    if (attr_ptr->key == ir::AttrType::ThreadExtent) {
      auto thread_tag = ir::ptr_cast<IterVar>(attr_ptr->node)->thread_tag;
      if (thread_extent_scope_.count(thread_tag)) {
        // TODO(ruobing): add this check
        // ELENA_ASSERT(is_zero(ir::Simplify(it->second - op->value)), "Should
        // Be zero.");
        return MutatorBase::visit(attr_ptr->body.get());
      } else {
        thread_extent_scope_[thread_tag] = attr_ptr->value;
        ir::NodePtr ret = MutatorBase::visit(attr_ptr);
        thread_extent_scope_.erase(thread_tag);
        return ret;
      }
    }
    return MutatorBase::visit(attr_ptr);
  }
  using MutatorBase::visit;

  /// Invoke mutate function.
  /// Typical Usage:
  /// \code
  ///   postProc(stmt);
  /// \encode
  ///
  /// \param body pointer to the instance of class Stmt;
  ///
  /// \return the instance of class Stmt
  /// which has no duplicated thread extent attr.
  StmtPtr postProc(StmtPtr body) {
    mutate(body);
    return body;
  }

 private:
  // The thread extent scope.
  std::unordered_map<std::string, ir::NodePtr> thread_extent_scope_;
};

StmtPtr scheduleToStatement(SchedulePtr sch, MapPtr<IterVar, Range> dom_map) {
  StmtPtr body = nullptr;
  int shared_scope_num = 0;
  int shared_scope_index = 0;
  for (auto i = sch->stages->size(); i != 0; --i) {
    StagePtr s = sch->stages->element[i - 1];
    if (s->scope.find("share") != s->scope.npos) {
      shared_scope_num++;
    }
  }
  for (auto i = sch->stages->size(); i != 0; --i) {
    StagePtr s = sch->stages->element[i - 1];
    if (s->scope.find("share") != s->scope.npos) {
      shared_scope_index++;
      if (shared_scope_index == 1) {
        s->sync_type = 1;
      } else if (shared_scope_index == shared_scope_num) {
        s->sync_type = 2;
      }
    }
  }

  // reverse the post DFS order.
  for (auto i = sch->stages->size(); i != 0; --i) {
    StagePtr s = sch->stages->element[i - 1];
    // no need to specify place holder op.
    if (s->op->get_type() == IRNodeType::PlaceholderOp) continue;
    // Remove grouping sugar, get the real attach spec.

    if (s->attach_type == ir::AttachType::InlinedAlready) {
      // do nothing
    } else if (s->attach_type == ir::AttachType::GroupRoot) {
      body = makePipeline(s, dom_map, body);
    } else if (s->attach_type == AttachType::Scope) {
      ELENA_ASSERT_EQ(s->attach_var->element.size(), 1,
                      "There should only be one attached itervar");
      auto attach_iter = s->attach_var->element[0];
      InjectAttach mutator(s, attach_iter, dom_map);
      body = mutator.inject_attach(body);
    } else {
      abort();
    }
  }
  SchedulePostProc post_proc;
  if (body == nullptr) {
    return body;
  }
  return post_proc.postProc(body);
}

void dump_code(std::string code, std::string file_name) {
  std::ofstream outfile(file_name, std::ios::ate);
  if (outfile.is_open()) {
    outfile << code << std::endl;
    outfile.close();
  }
}

void dump_diminfo(int bx, int by, int bz, int tx, int ty, int tz) {
  std::ofstream outfile("schedule.txt", std::ios::ate);
  if (outfile.is_open()) {
    outfile << bx << std::endl;
    outfile << by << std::endl;
    outfile << bz << std::endl;
    outfile << tx << std::endl;
    outfile << ty << std::endl;
    outfile << tz << std::endl;
    outfile.close();
  }
}
}  // namespace api
