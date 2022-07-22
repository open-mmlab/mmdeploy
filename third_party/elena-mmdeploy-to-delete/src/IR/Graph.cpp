#include "IR/Graph.h"

#include <algorithm>

#include "IR/ExprVisitor.h"

namespace graph {

// Map<Operation, Array<IterVar>>;
std::unordered_map<ir::OpPtr, std::vector<ir::IterVarPtr>> CreateAttachPath(
    ir::SchedulePtr sch) {
  std::unordered_map<ir::OpPtr, std::vector<ir::IterVarPtr>> ret;
  for (auto stage : sch->stages->element) {
    std::vector<ir::IterVarPtr> path;
    for (auto s = stage; s != nullptr;) {
      auto spec = s->get_attach_spec();
      bool start_attach;
      ir::IterVarPtr attach_ivar;
      if (spec->attach_type == ir::AttachType::Scope) {
        attach_ivar = spec->attach_var->element[0];
        s = spec->attach_stage->element[0];
        start_attach = false;
      } else {
        break;
      }
      for (size_t i = s->leaf_itervars->element.size(); i != 0; --i) {
        auto iv = s->leaf_itervars->element[i - 1];
        if (!start_attach && iv == attach_ivar) {
          start_attach = true;
        }
        if (start_attach) path.push_back(iv);
      }
    }
    if (!ret.count(stage->op)) {
      ret[stage->op] = path;
    }
  }
  return ret;
}

ReadGraph CreateReadGraph(const ir::ArrayPtr<ir::Op>& poi) {
  ReadGraph ret;
  ir::ArrayPtr<ir::Op> stack = std::make_shared<ir::Array<ir::Op>>();
  std::unordered_set<ir::OpPtr> vis;
  std::copy(poi->element.begin(), poi->element.end(),
            std::back_inserter(stack->element));
  vis.insert(poi->element.begin(), poi->element.end());
  while (!stack->element.empty()) {
    ir::OpPtr op = stack->element.back();
    stack->element.pop_back();
    ir::ArrayPtr<ir::TensorVar> fas =
        std::make_shared<ir::Array<ir::TensorVar>>();

    if (op->get_type() == ir::IRNodeType::ComputeOp) {
      ir::ComputeOpPtr cop = ir::ptr_cast<ir::ComputeOp>(op);
      ExprVisitor ev;
      fas = ev.tensorInExpr(cop->fcompute);
    }

    ret[op] = fas;
    for (auto i : fas->element) {
      if (i->op != nullptr && vis.count(i->op) == 0) {
        vis.insert(i->op);
        stack->element.push_back(i->op);
      }
    }
  }

  return ret;
}

FeedGraph CreateFeedGraph(const ReadGraph& g) {
  FeedGraph fg;
  for (auto kv : g) {
    for (auto t : kv.second->element) {
      if (!fg.count(t)) {
        fg[t] = std::make_shared<ir::Array<ir::Op>>();
      }
      fg[t]->element.push_back(kv.first);
    }
  }
  return fg;
}

void DoGraphDfs(
    ir::OpPtr i,
    const std::unordered_map<ir::OpPtr, ir::ArrayPtr<ir::TensorVar>>& graph,
    std::unordered_set<ir::OpPtr>* vis, ir::ArrayPtr<ir::Op> ret) {
  if (vis->count(i)) return;
  vis->insert(i);
  for (const auto& j : graph.at(i)->element) {
    DoGraphDfs(j->op, graph, vis, ret);
  }
  ret->element.push_back(i);
}

ir::ArrayPtr<ir::Op> GraphDfs(
    const ir::ArrayPtr<ir::Op>& poi,
    const std::unordered_map<ir::OpPtr, ir::ArrayPtr<ir::TensorVar>>& graph) {
  std::unordered_set<ir::OpPtr> vis;
  ir::ArrayPtr<ir::Op> ret = std::make_shared<ir::Array<ir::Op>>();
  for (ir::OpPtr i : poi->element) {
    DoGraphDfs(i, graph, &vis, ret);
  }

  return ret;
}

bool GetSubGraphDfs(const ir::OpPtr& t,
                    const std::unordered_set<ir::OpPtr>& lim, bool includein,
                    std::unordered_map<ir::OpPtr, bool>* vis,
                    ir::ArrayPtr<ir::Op> ret) {
  if (vis->count(t)) {
    return vis->at(t);
  }
  if (lim.count(t)) {
    (*vis)[t] = true;
    if (includein) {
      ret->element.push_back(t);
    }
    return true;
  }
  (*vis)[t] = false;
  bool tmpv = false;

  if (t->get_type() == ir::IRNodeType::ComputeOp) {
    ir::ComputeOpPtr cop = ir::ptr_cast<ir::ComputeOp>(t);
    ExprVisitor ev;
    ir::ArrayPtr<ir::TensorVar> ipt = ev.tensorInExpr(cop->fcompute);
    for (auto i : ipt->element) {
      if (GetSubGraphDfs(i->op, lim, includein, vis, ret)) {
        tmpv = true;
      }
    }
  }
  if (tmpv == true) {
    (*vis)[t] = true;
    ret->element.push_back(t);
  }
  return tmpv;
}

ir::ArrayPtr<ir::Op> GetSubGraph(const ir::ArrayPtr<ir::TensorVar>& outputs,
                                 const ir::ArrayPtr<ir::TensorVar>& inputs,
                                 bool includein) {
  ir::ArrayPtr<ir::Op> ret = std::make_shared<ir::Array<ir::Op>>();
  std::unordered_set<ir::OpPtr> lim;
  std::unordered_map<ir::OpPtr, bool> vis;
  for (auto i : inputs->element) {
    lim.insert(i->op);
  }
  for (ir::TensorVarPtr i : outputs->element) {
    GetSubGraphDfs(i->op, lim, includein, &vis, ret);
  }
  return ret;
}

ir::StagePtr GroupLCA(ir::StagePtr p1, ir::StagePtr p2) {
  if (p1 == nullptr) return p1;
  if (p2 == nullptr) return p2;
  if (p1 == p2) return p1;
  ir::StagePtr fa = p1;
  while (fa != nullptr) {
    if (fa == p2) return p2;
    fa = fa->group;
  }
  fa = p2;
  while (fa != nullptr) {
    if (fa == p1) return p1;
    fa = fa->group;
  }
  return fa;
}

}  // namespace graph
