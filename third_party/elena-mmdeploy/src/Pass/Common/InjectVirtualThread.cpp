#include "Pass/Common/InjectVirtualThread.h"

#include <vector>

#include "IR/Expr.h"
#include "IR/Type.h"
#include "Pass/Common/StmtCopy.h"
#include "api.h"
using ir::Node;

ir::StmtPtr substitute(ir::StmtPtr stmt,
                       std::unordered_map<std::string, ir::ExprPtr> value_map) {
  if (value_map.size() == 0) return stmt;
  return ir::ptr_cast<ir::Stmt>(IRSubstitue(value_map).visit(stmt.get()));
}

class FindVarInExpr : public VisitorBase<FindVarInExpr> {
 public:
  using VisitorBase::visit;
  void visit(ir::TensorVar* node) {
    if (vars.count(node->get_name())) {
      find_var = true;
      return;
    }
  }
  void visit(ir::IterVar* node) {
    if (vars.count(node->get_name())) {
      find_var = true;
      return;
    }
  }
  bool checkVar(ExprPtr body, std::unordered_set<std::string> vars_) {
    vars = vars_;
    find_var = false;
    visit(body);
    return find_var;
  }

 private:
  std::unordered_set<std::string> vars;
  bool find_var;
};

class FindRealtiveVar : public VisitorBase<FindRealtiveVar> {
 public:
  using VisitorBase::visit;
  void visit(ir::Store* node) {
    auto store_ptr = node->shared_from_this();
    visit(store_ptr->value);
    if (var_finder.checkVar(store_ptr->value, touch_set)) {
      touch_set.insert(store_ptr->var->get_name());
    }
  }
  void visit(ir::For* node) {
    auto for_ptr = node->shared_from_this();
    if (var_finder.checkVar(for_ptr->init, touch_set)) {
      touch_set.insert(for_ptr->it->get_name());
    }
    visit(for_ptr->body);
  }
  void visit(ir::Let* node) {
    auto let_ptr = node->shared_from_this();
    if (var_finder.checkVar(let_ptr->value, touch_set)) {
      touch_set.insert(let_ptr->var->get_name());
    }
    visit(let_ptr->body);
  }
  std::unordered_set<std::string> getRelativeVar(StmtPtr body,
                                                 NodePtr virtual_var) {
    auto ptr = ir::ptr_cast<ir::IterVar>(virtual_var);
    touch_set.insert(ptr->get_name());
    visit(body);
    return touch_set;
  }

 private:
  std::unordered_set<std::string> touch_set;
  FindVarInExpr var_finder;
};

class VTInjector : public MutatorBase<VTInjector> {
 public:
  using MutatorBase::visit;
  VTInjector(ir::IterVarPtr vthread_var_, ir::ExprPtr vthread_extent_,
             std::unordered_set<std::string> relative_var_set_)
      : vthread_var(vthread_var_),
        vthread_extent(vthread_extent_),
        relative_var_set(relative_var_set_),
        injecting_vthread(false) {}

  NodePtr visit(ir::IfThenElse* node) {
    auto if_ptr = node->shared_from_this();
    if (var_finder.checkVar(if_ptr->condition, relative_var_set) &&
        !injecting_vthread) {
      // need inject vthread above this IfThenElse stmt
      injecting_vthread = true;
      mutate(if_ptr->then_case);
      if (if_ptr->else_case) {
        mutate(if_ptr->else_case);
      }
      injecting_vthread = false;
      return injectVTLoop(if_ptr);
    } else {
      mutate(if_ptr->then_case);
      if (if_ptr->else_case) {
        mutate(if_ptr->else_case);
      }
      return if_ptr;
    }
  }
  NodePtr visit(ir::Let* node) {
    auto let_ptr = node->shared_from_this();
    if (var_finder.checkVar(let_ptr->var, relative_var_set) &&
        !injecting_vthread) {
      // need inject vthread above this IfThenElse stmt
      injecting_vthread = true;
      mutate(let_ptr->body);
      injecting_vthread = false;
      return injectVTLoop(let_ptr);
    } else {
      mutate(let_ptr->body);
      return let_ptr;
    }
  }
  NodePtr visit(ir::For* node) {
    auto for_ptr = node->shared_from_this();
    if ((var_finder.checkVar(for_ptr->init, relative_var_set) ||
         var_finder.checkVar(for_ptr->extent, relative_var_set)) &&
        !injecting_vthread) {
      // need inject vthread above this IfThenElse stmt
      injecting_vthread = true;
      mutate(for_ptr->body);
      injecting_vthread = false;
      return injectVTLoop(for_ptr);
    } else {
      mutate(for_ptr->body);
      return for_ptr;
    }
  }
  NodePtr visit(ir::Allocate* node) {
    auto allocate_ptr = node->shared_from_this();
    if (var_finder.checkVar(allocate_ptr->var, relative_var_set)) {
      // CHECK there has only one element in bound
      new_size[allocate_ptr->var] = allocate_ptr->bound->element[0]->extent;
      allocate_ptr->bound->element[0] = std::make_shared<ir::Range>(
          api::constant<uint64_t>(0),
          vthread_extent * allocate_ptr->bound->element[0]->extent);
    }
    mutate(allocate_ptr->body);
    return allocate_ptr;
  }

  NodePtr visit(ir::Store* node) {
    auto store_ptr = node->shared_from_this();
    if (var_finder.checkVar(store_ptr->var, relative_var_set)) {
      // CHECK there has only one element in bound
      if (new_size.count(store_ptr->var)) {
        store_ptr->index->element[0] = store_ptr->index->element[0] +
                                       new_size[store_ptr->var] * vthread_var;
      }
      mutate(store_ptr->value);
      if (!injecting_vthread) {
        return injectVTLoop(store_ptr);
      } else {
        return store_ptr;
      }
    }
    return store_ptr;
  }
  NodePtr visit(ir::ScalarVar* node) {
    auto scalar_ptr = node->shared_from_this();
    if (var_finder.checkVar(scalar_ptr->tensor, relative_var_set) &&
        new_size.count(scalar_ptr->tensor)) {
      // CHECK there has only one element in bound
      scalar_ptr->indices->element[0] =
          scalar_ptr->indices->element[0] +
          new_size[scalar_ptr->tensor] * vthread_var;
    }
    return scalar_ptr;
  }

  // inject vthread loop
  StmtPtr injectVTLoop(StmtPtr stmt) {
    auto num_threads = ir::ptr_cast<Const<uint64_t>>(vthread_extent);
    ELENA_ASSERT(num_threads, "vthread_extent is not const");
    int num_threads_ = num_threads->get_value();
    if (num_threads_ < 16) {
      // do unrolling if it is inside innermost content.
      std::unordered_map<std::string, ir::ExprPtr> value_map;
      StmtPtr blk_ = nullptr;
      for (int i = 1; i < num_threads_; i++) {
        value_map[vthread_var->get_name()] =
            std::make_shared<Const<uint64_t>>(i, ir::ScalarType::UInt64);
        StmtPtr blk = ir::ptr_cast<ir::Stmt>(stmt_copyer.stmt_copy(stmt));
        blk = substitute(blk, value_map);
        if (blk_)
          blk_ = std::make_shared<ir::Block>(blk_, blk);
        else
          blk_ = blk;
      }
      value_map[vthread_var->get_name()] =
          std::make_shared<Const<uint64_t>>(0, ir::ScalarType::UInt64);
      stmt = substitute(stmt, value_map);
      blk_ = std::make_shared<ir::Block>(stmt, blk_);
      return blk_;
    } else {
      // insert a for loop
      return std::make_shared<ir::For>(
          vthread_var,
          std::make_shared<Const<uint64_t>>(0, ir::ScalarType::UInt64),
          vthread_extent, stmt);
    }
  }

  ir::StmtPtr injectVthread(StmtPtr body) {
    mutate(body);
    return body;
  }

 private:
  std::unordered_map<ir::VarPtr, ir::ExprPtr> new_size;
  std::unordered_set<std::string> relative_var_set;
  ir::IterVarPtr vthread_var;
  ir::ExprPtr vthread_extent;
  FindVarInExpr var_finder;
  // whethe the loop is already injected.
  bool injecting_vthread;
  StmtCopy stmt_copyer;
};

VirtualThreadInjector::VirtualThreadInjector() {}

NodePtr VirtualThreadInjector::mutateReplace(NodePtr node) {
  mutate(node);
  return node;
}

NodePtr VirtualThreadInjector::visit(ir::Attr* node) {
  CHECK_NODE_TYPE(node, Attr);
  auto attr = node->shared_from_this();
  auto key = attr->key;
  if (key == ir::AttrType::VirtualThread) {
    auto thread_tag = ir::ptr_cast<ir::IterVar>(attr->node)->thread_tag;
    if (thread_tag.find("vthread") != thread_tag.npos) {  // is virtual thread
      FindRealtiveVar relative_var_finder;
      auto relative_var =
          relative_var_finder.getRelativeVar(attr->body, attr->node);
      VTInjector injector(ir::ptr_cast<ir::IterVar>(attr->node),
                          ir::ptr_cast<ir::Expr>(attr->value), relative_var);
      attr->body = injector.injectVthread(attr->body);
      mutate(attr->body);
      return attr;
    } else {
      mutate(attr->body);
      return node->shared_from_this();
    }
  } else {
    mutate(attr->body);
    return node->shared_from_this();
  }
}

namespace api {
ir::StmtPtr injectVirtualThread(ir::StmtPtr stmt) {
  auto inject_vthread = std::make_shared<VirtualThreadInjector>();
  auto injected_node = inject_vthread->mutateReplace(stmt);
  auto injected_stmt = ir::ptr_cast<ir::Stmt, ir::Node>(injected_node);
  return injected_stmt;
}

}  // namespace api
