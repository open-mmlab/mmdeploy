#include "Pass/Common/HoistIfThenElse.h"

class IRApplyVisit : public VisitorBase<IRApplyVisit> {
 public:
  explicit IRApplyVisit(std::function<void(Node*)> Fvisit) : Func(Fvisit) {}

  using VisitorBase::visit;
  void visit(Node* node) {
    if (Visited.count(node) != 0) return;
    Visited.insert(node);
    VisitorBase::visit(node);
    Func(node);
  }

 private:
  std::function<void(Node*)> Func;
  std::unordered_set<const Node*> Visited;
};

void postOrderVisit(NodePtr node, std::function<void(Node*)> Fvisit) {
  IRApplyVisit(Fvisit).visit(node.get());
}

NodePtr getShareptr(Node* node) {
  ir::IRNodeType node_type = node->get_type();
  switch (node_type) {
#define IR_NODE_TYPE_PLAIN(Type)                  \
  case ir::IRNodeType::Type: {                    \
    ir::Type* ptr = static_cast<ir::Type*>(node); \
    return ptr->shared_from_this();               \
  }
#define IR_NODE_TYPE_ABSTRACT(Type) \
  case ir::IRNodeType::Type: {      \
    return nullptr;                 \
  }
#define IR_NODE_TYPE_NESTED(Type) \
  case ir::IRNodeType::Type: {    \
    return nullptr;               \
  }
#include "x/ir_node_types.def"
  }
  return nullptr;
}

class IRApplyMutate : public MutatorBase<IRApplyMutate> {
 public:
  explicit IRApplyMutate(std::function<NodePtr(NodePtr)> Fvisit)
      : Func(Fvisit) {}

  using MutatorBase::visit;
  NodePtr visit(ir::For* node) {
    if (Visited.count(node) != 0) return node->shared_from_this();
    Visited.insert(node);
    auto ptr = MutatorBase::visit(node);
    if (ptr->get_type() == ir::IRNodeType::For) {
      auto result = Func(ptr);
      NodePtr if_ptr;
      if (result->get_type() == ir::IRNodeType::IfThenElse) {
        auto if_ptr_ = ir::ptr_cast<ir::IfThenElse>(result);
        if_ptr = if_ptr_->shared_from_this();
      } else {
        if_ptr = result;
      }
      auto stmt = ir::ptr_cast<ir::Stmt>(if_ptr);
      return if_ptr;
    }
    return node->shared_from_this();
  }

 private:
  std::function<NodePtr(NodePtr)> Func;
  std::unordered_set<const Node*> Visited;
};

NodePtr postOrderMutate(NodePtr node, std::function<NodePtr(NodePtr)> Fmutate) {
  return IRApplyMutate(Fmutate).visit(node.get());
}

class IRApplyMutateIf : public MutatorBase<IRApplyMutateIf> {
 public:
  explicit IRApplyMutateIf(std::function<Node*(Node*)> f) : f_(f) {}

  using MutatorBase::visit;
  NodePtr visit(ir::IfThenElse* node) {
    if (visited_.count(node) != 0) return node->shared_from_this();
    visited_.insert(node);
    auto ptr = MutatorBase::visit(node);
    if (ptr->get_type() == ir::IRNodeType::IfThenElse) {
      auto result = f_(ptr.get());
      auto if_ptr = getShareptr(result);
      auto stmt = ir::ptr_cast<ir::Stmt>(if_ptr);
      return if_ptr;
    }
    return node->shared_from_this();
  }

 private:
  std::function<Node*(Node*)> f_;
  std::unordered_set<const Node*> visited_;
};

NodePtr postOrderMutateIf(NodePtr node, std::function<Node*(Node*)> Fmutate) {
  return IRApplyMutateIf(Fmutate).visit(node.get());
}

// Locate all For nodes and capture child IfThenElse nodes.
void IfThenElseHoist::selectCandidates(StmtPtr stmt) {
  postOrderVisit(stmt, [&](Node* node) {
    if (node->get_type() != ir::IRNodeType::For) return;
    auto for_ptr = static_cast<ir::For*>(node);
    std::queue<StmtPtr> tracker;
    tracker.push(for_ptr->body);
    For2IfMap.insert({for_ptr, std::vector<StmtPtr>()});
    while (!tracker.empty()) {
      StmtPtr head = tracker.front();
      tracker.pop();
      if (head->get_type() == ir::IRNodeType::For) {
        for (auto if_ptr : For2IfMap.at(head.get())) {
          For2IfMap[for_ptr].push_back(if_ptr);
        }
      } else if (head->get_type() == ir::IRNodeType::Attr) {
        auto attr_ptr = ir::ptr_cast<ir::Attr>(head);
        tracker.push(attr_ptr->body);
      } else if (head->get_type() == ir::IRNodeType::IfThenElse) {
        For2IfMap[for_ptr].push_back(head);
        auto if_ptr = ir::ptr_cast<ir::IfThenElse>(head);
        tracker.push(if_ptr->then_case);
        if (if_ptr->else_case) {
          tracker.push(if_ptr->else_case);
        }
        if (!CondVarMap.count(head.get())) {
          std::unordered_set<Node*> new_var_set;
          CondVarMap.insert({head.get(), new_var_set});
          postOrderVisit(if_ptr->condition, [&](Node* cond_node) {
            if (cond_node->get_type() == ir::IRNodeType::IterVar) {
              CondVarMap[head.get()].insert(cond_node);
            }
          });
        }
      } else {
        continue;
      }
    }
    OrderedForList.emplace_back(static_cast<ir::Stmt*>(node));
  });
}

// For each IfThenElse node, find the highest For node which
// meets loop invariant condition.
void IfThenElseHoist::locateTopFor() {
  std::unordered_map<Node*, ir::StmtPtr> if_position_map;
  std::unordered_set<Node*> top_for_var_set;

  // Create IfThenElse -> For map.
  for (auto for_stmt : OrderedForList) {
    std::vector<ir::StmtPtr> if_list = For2IfMap[for_stmt];
    if (for_stmt->get_type() != ir::IRNodeType::For) return;
    if (if_list.empty()) continue;
    auto for_node = static_cast<ir::For*>(for_stmt);
    ELENA_ASSERT(for_node, "Failed to static_cast to For node");
    TopForVarMap.insert({for_node->it.get(), if_list});
    for (auto if_stmt : if_list) {
      Node* if_node = if_stmt.get();
      If2ForMap[if_node].push_back(for_node->shared_from_this());
    }
  }

  // Locate the highest For node which is loop invariant.
  for (auto item : If2ForMap) {
    ir::StmtPtr top_for = nullptr;
    Node* if_stmt = item.first;
    std::vector<ir::StmtPtr> for_list = item.second;
    for (size_t i = 0; i < for_list.size(); ++i) {
      const ir::StmtPtr for_stmt = for_list.at(i);
      auto for_node = ir::ptr_cast<ir::For>(for_stmt);
      ELENA_ASSERT(for_node, "Failed to ptr_cast to For node");
      std::vector<ir::StmtPtr> new_for_list{for_stmt};
      ForTrackingMap.insert({for_stmt.get(), new_for_list});
      if (CondVarMap[if_stmt].count(for_node->it.get())) {
        std::vector<ir::StmtPtr> updated_for_list(for_list.begin(),
                                                  for_list.begin() + i);
        If2ForMap[if_stmt] = updated_for_list;
        break;
      } else {
        top_for = for_stmt;
      }
    }
    if (top_for) {
      if (top_for->get_type() == ir::IRNodeType::For) {
        auto top_for_node = ir::ptr_cast<ir::For>(top_for);
        if_position_map.insert({if_stmt, top_for_node});
      }
    }
  }

  for (auto item : if_position_map) {
    auto for_node = ir::ptr_cast<ir::For>(item.second);
    top_for_var_set.insert(for_node->it.get());
  }

  std::vector<Node*> removed_for_var_list;
  for (auto item : TopForVarMap) {
    Node* top_for_var = item.first;
    std::vector<StmtPtr> if_list = item.second;
    if (!top_for_var_set.count(top_for_var)) {
      removed_for_var_list.push_back(top_for_var);
    } else {
      std::vector<StmtPtr> actual_if_list;
      for (auto if_stmt : if_list) {
        if (if_position_map.count(if_stmt.get())) {
          actual_if_list.push_back(if_stmt);
        }
      }
      TopForVarMap[top_for_var] = actual_if_list;
    }
  }
  for (auto top_for_var : removed_for_var_list) {
    TopForVarMap.erase(top_for_var);
  }
}

// Remove IfThenElse node from a For node.
// A pair of For nodes will be generated.
std::pair<ir::StmtPtr, ir::StmtPtr> RemoveIf(StmtPtr for_stmt,
                                             StmtPtr if_stmt) {
  StmtPtr then_for;
  StmtPtr else_for;

  auto if_ptr = ir::ptr_cast<ir::IfThenElse>(if_stmt);
  ELENA_ASSERT(if_ptr, "Failed to Cast to IfThenElse node");

  then_for =
      ir::ptr_cast<ir::Stmt>(postOrderMutateIf(for_stmt, [&](Node* node) {
        if (node != if_stmt.get()) return node;
        if (node == if_stmt.get()) {
          auto ptr = static_cast<ir::IfThenElse*>(node);
          ELENA_ASSERT(ptr, "Failed to Cast to IfThenElse node");
          return reinterpret_cast<Node*>(ptr->then_case.get());
        }
      }));

  if (if_ptr->else_case) {
    auto else_for =
        ir::ptr_cast<ir::Stmt>(postOrderMutateIf(for_stmt, [&](Node* node) {
          if (node != if_stmt.get()) return node;
          if (node == if_stmt.get()) {
            auto ptr = static_cast<ir::IfThenElse*>(node);
            ELENA_ASSERT(ptr, "Failed to Cast to IfThenElse node");
            return reinterpret_cast<Node*>(ptr->else_case.get());
          }
        }));
  }
  return std::make_pair(then_for, else_for);
}

// ASSERT whether a given IfThenElse stmt is the first one appearing
// in a For stmt.
bool is_first_if(StmtPtr for_stmt, StmtPtr if_stmt) {
  std::vector<const Node*> if_node_list;

  auto if_node = ir::ptr_cast<ir::IfThenElse>(if_stmt);
  ELENA_ASSERT(if_node, "Failed to Cast to IfThenElse node")

  postOrderVisit(for_stmt, [&](Node* node) {
    if (node->get_type() == ir::IRNodeType::IfThenElse) {
      if_node_list.push_back(node);
    }
  });
  return if_node_list.empty() ? false : if_stmt.get() == if_node_list.back();
}

// When we try to mutate a For node, some child For nodes can have already
// been mutated. This function is to get the updated For node and further
// hoisting can be done based on this new node.
// We keep all For nodes tracing in for_tracking_map_. When we get a
// hoisted IfThenElse, we match it with tracing For nodes to pick
// the updated one.
size_t IfThenElseHoist::getUpdatedFor(StmtPtr for_stmt, StmtPtr if_stmt) {
  std::vector<StmtPtr> tracked_for_list = ForTrackingMap[for_stmt.get()];
  size_t updated_for_idx = 0;
  for (size_t i = 0; i < tracked_for_list.size(); ++i) {
    StmtPtr current_for = tracked_for_list.at(tracked_for_list.size() - 1 - i);
    if (is_first_if(current_for, if_stmt)) {
      updated_for_idx = tracked_for_list.size() - 1 - i;
      break;
    }
  }
  return updated_for_idx;
}

// Update upper level For node when current For node is modified.
// With this function we only need to visit and mutate top level For node
// in the main VisitAndMutate function.
StmtPtr update_for(StmtPtr parent_for_stmt, StmtPtr new_if_stmt) {
  const Node* top_for_node;
  auto parent_for_node = ir::ptr_cast<ir::For>(parent_for_stmt);
  ELENA_ASSERT(parent_for_node, "Failed to Cast to For node")

  auto new_if_node = ir::ptr_cast<ir::IfThenElse>(new_if_stmt);
  ELENA_ASSERT(new_if_node, "Failed to Cast to IfThenElse node")

  postOrderVisit(parent_for_node->body, [&](Node* node) {
    if (node->get_type() == ir::IRNodeType::For) {
      top_for_node = node;
    }
  });

  return ir::ptr_cast<ir::Stmt>(
      postOrderMutate(parent_for_stmt, [&](NodePtr node) {
        if (node.get() == top_for_node) {
          return (NodePtr)new_if_stmt;
        }
        return node;
      }));
}

// Hoist an IfThenElse node as high as possible.
// This function iterates on all candidate For nodes. For each For node,
// it first removes IfThenElse nodes. Then it generates a new IfThenElse
// node using mutated For nodes.
StmtPtr IfThenElseHoist::hoistIf(StmtPtr if_stmt) {
  StmtPtr new_if = if_stmt;

  for (size_t i = 0; i < If2ForMap[if_stmt.get()].size(); ++i) {
    new_if = if_stmt;
    StmtPtr for_stmt = If2ForMap[if_stmt.get()].at(i);
    size_t updated_for_idx = getUpdatedFor(for_stmt, new_if);
    StmtPtr updated_for_node =
        ForTrackingMap[for_stmt.get()].at(updated_for_idx);
    auto generated_for_pair = RemoveIf(updated_for_node, new_if);
    StmtPtr then_for = generated_for_pair.first;
    StmtPtr else_for = generated_for_pair.second;
    ForTrackingMap[for_stmt.get()].at(updated_for_idx) = then_for;

    if (else_for.get()) {
      ForTrackingMap[for_stmt.get()].push_back(else_for);
    }

    auto new_if_node = ir::ptr_cast<ir::IfThenElse>(new_if);
    ELENA_ASSERT(new_if_node, "Failed to Cast to IfThenElse node");

    new_if_node->then_case = then_for;
    new_if_node->else_case = else_for;

    if (i < If2ForMap[if_stmt.get()].size() - 1) {
      StmtPtr original_next_for = If2ForMap[if_stmt.get()].at(i + 1);
      StmtPtr actual_next_for =
          ForTrackingMap[original_next_for.get()].at(updated_for_idx);
      StmtPtr update_for_stmt = update_for(actual_next_for, new_if);
      ForTrackingMap[original_next_for.get()].at(updated_for_idx) =
          update_for_stmt;
    }
  }
  return new_if;
}

StmtPtr IfThenElseHoist::postOrderMutateStmt(StmtPtr stmt) {
  auto stmt_ptr = postOrderMutate(stmt, [&](NodePtr node) {
    StmtPtr ret;
    if (node->get_type() != ir::IRNodeType::For) return node;
    auto ptr = ir::ptr_cast<ir::For>(node);
    ELENA_ASSERT(ptr, "Failed to Cast to For node");
    ir::ForPtr for_node = ptr->shared_from_this();
    if (TopForVarMap.count(for_node->it.get())) {
      std::vector<StmtPtr> new_if_list;
      for (auto if_stmt : TopForVarMap[for_node->it.get()]) {
        new_if_list.emplace_back(hoistIf(if_stmt));
      }
      ir::IfThenElsePtr next_if_node_;
      ELENA_ASSERT(new_if_list.back(), "new_if_list.back() is Null");
      auto current_if_node = ir::ptr_cast<ir::IfThenElse>(new_if_list.back());
      StmtPtr new_for;
      for (size_t i = new_if_list.size() - 1; i > 0; --i) {
        ELENA_ASSERT(current_if_node, "Failed to Cast to IfThenElse node");
        StmtPtr current_if_stmt = std::make_shared<ir::IfThenElse>(
            current_if_node->condition, current_if_node->then_case,
            current_if_node->else_case);

        next_if_node_ = ir::ptr_cast<ir::IfThenElse>(new_if_list[i - 1]);
        auto next_if_node = std::make_shared<ir::IfThenElse>(
            next_if_node_->condition, next_if_node_->then_case,
            next_if_node_->else_case);
        ELENA_ASSERT(next_if_node, "Failed to Cast to IfThenElse node");
        new_for = std::make_shared<ir::IfThenElse>(
            next_if_node->condition, current_if_stmt, next_if_node->else_case);
        current_if_node = ir::ptr_cast<ir::IfThenElse>(new_for);
      }

      if (!new_for.get()) {
        ir::IfThenElsePtr first_if_node =
            ir::ptr_cast<ir::IfThenElse>(new_if_list[0]);
        ELENA_ASSERT(first_if_node, "Failed to Cast to IfThenElse node");
        new_for = std::make_shared<ir::IfThenElse>(first_if_node->condition,
                                                   first_if_node->then_case,
                                                   first_if_node->else_case);
      }
      ret = new_for;
      return (NodePtr)ret;
    }
    return node;
  });
  return ir::ptr_cast<ir::Stmt>(stmt_ptr);
}

namespace api {
ir::StmtPtr hoistIfThenElse(ir::StmtPtr stmt) {
  IfThenElseHoist hoist_if;
  auto hoist_if_node = hoist_if.visitAndMutate(stmt);
  return hoist_if_node;
}

}  // namespace api
