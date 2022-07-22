#include "Schedule/Schedule.h"

#include "IR/ExprVisitor.h"
#include "IR/Graph.h"
#include "IR/InlineMutate.h"
#include "IR/VarReplacer.h"
#include "Schedule/TensorReplacer.h"
#include "api.h"
#include "logging.h"

namespace ir {

Schedule::Schedule() : Node(IRNodeType::Schedule) {
  outputs = std::make_shared<Array<Op>>();
  stages = std::make_shared<Array<Stage>>();
  groups = std::make_shared<Array<Stage>>();
  op_stage = std::make_shared<Map<Op, Stage>>();
}

Schedule::Schedule(
    ArrayPtr<Op> oparr,
    std::unordered_map<ir::OpPtr, std::vector<ir::OpPtr>>* shadow)
    : Node(IRNodeType::Schedule) {
  outputs = oparr;
  stages = std::make_shared<Array<Stage>>();
  groups = std::make_shared<Array<Stage>>();
  op_stage = std::make_shared<Map<Op, Stage>>();

  read_graph = graph::CreateReadGraph(outputs);
  feed_graph = graph::CreateFeedGraph(read_graph);

  ArrayPtr<Op> order = graph::GraphDfs(outputs, read_graph);
  std::unordered_set<OpPtr> output_set;
  output_set.insert(oparr->element.begin(), oparr->element.end());
  for (OpPtr op : order->element) {
    StagePtr tmpstage = std::make_shared<Stage>(op);
    tmpstage->is_output = (output_set.count(op) != 0);
    stages->element.push_back(tmpstage);
    op_stage->element[op] = tmpstage;
  }
}

StagePtr Schedule::operator[](const OpPtr& op) {
  auto it = op_stage->element.find(op);
  if (it == op_stage->element.end()) {
    return nullptr;
  }
  ELENA_ASSERT_NE(it, op_stage->element.end(),
                  "Can't find Stage for this op in the Schedule");
  return (*it).second;
}

StagePtr Schedule::operator[](const TensorVarPtr& tensor) {
  return operator[](tensor->op);
}

void Schedule::InvalidateCache() { cache.clear(); }

void Schedule::InitCache() {
  if (stages->element.size() == cache.size()) return;
  InvalidateCache();
  for (StagePtr i : stages->element) {
    cache[i->op] = i;
  }
}

// not used temporarily. ScanOp related, lay aside.
StagePtr Schedule::create_group(const ArrayPtr<TensorVar>& outputs,
                                const ArrayPtr<TensorVar>& inputs,
                                bool includein) {
  InitCache();
  const auto& scache = cache;
  ArrayPtr<Op> ops = graph::GetSubGraph(outputs, inputs, includein);

  std::unordered_map<StagePtr, int> cnt;
  StagePtr father_group;
  for (size_t i = 0; i < ops->element.size(); ++i) {
    OpPtr iop = (*ops)[i];
    auto it = scache.find(iop);
    StagePtr istage = (*it).second->group;
    if (i == 0) {
      father_group = istage;
    } else {
      father_group = graph::GroupLCA(father_group, istage);
    }
    if (istage != nullptr) {
      ++cnt[istage];
    }
  }
  StagePtr gstage = std::make_shared<Stage>();
  gstage->group = father_group;
  if (father_group != nullptr) {
    ++father_group->num_child_stages;
  }
  std::vector<StagePtr> stack;
  for (auto& ki : cnt) {
    if ((ki.first != father_group) &&
        (ki.first->num_child_stages == ki.second)) {
      stack.push_back(ki.first);
    }
  }
  while (!stack.empty()) {
    StagePtr g = stack.back();
    stack.pop_back();
    if ((g->group != nullptr) && (g->group != father_group)) {
      int tmp = ++cnt[g->group];
      if (tmp == g->group->num_child_stages) {
        stack.push_back(g->group);
      }
    }
  }
  for (auto& ki : cnt) {
    if (ki.first == father_group) continue;
    ELENA_ASSERT_EQ(
        ki.first->num_child_stages, ki.second,
        "Trying to group region that intersect with an already existed group");
    if (ki.first->group == father_group) {
      StagePtr s = ki.first;
      s->group = gstage;
      ++gstage->num_child_stages;
      if (father_group != nullptr) {
        --father_group->num_child_stages;
      }
    }
  }
  for (OpPtr op : ops->element) {
    auto it = scache.find(op);
    StagePtr s = it->second;
    if (s->group == father_group) {
      s->group = gstage;
      ++gstage->num_child_stages;
      if (father_group != nullptr) {
        --father_group->num_child_stages;
      }
    }
  }
  for (OpPtr op : ops->element) {
    auto it = scache.find(op);
    StagePtr s = it->second;
    if (s->attach_type == AttachType::Scope) {
      StagePtr cg;
      for (size_t i = 0; i < s->attach_stage->size(); ++i) {
        cg = graph::GroupLCA(s->attach_stage->element[i]->group, gstage);
      }
      if (cg != gstage) {
        s->compute_root();
      }
    }
  }
  groups->element.push_back(gstage);
  return gstage;
}

ExprPtr ReplaceTensor(ExprPtr expr, const TensorVarMap<TensorVarPtr>& rmap) {
  TensorReplacer tr(rmap);
  return ir::ptr_cast<Expr>(tr.MutateReplace(expr));
}

OpPtr ReplaceInputs(const OpPtr& op, const TensorVarMap<TensorVarPtr>& rmap) {
  if (op->get_type() == IRNodeType::ComputeOp) {
    ComputeOpPtr cop = ir::ptr_cast<ComputeOp>(op);
    ExprPtr new_fcompute;
    new_fcompute = ReplaceTensor(cop->fcompute, rmap);
    if (new_fcompute != cop->fcompute) {
      return ComputeOp::create(cop->output_shape(0), cop->iter_vars,
                               new_fcompute, cop->output(0)->get_name());
    } else {
      return op;
    }
  }
  if (op->get_type() == IRNodeType::PlaceholderOp) {
    return op;
  }
  return op;
}

void ReplaceDataFlow(const ArrayPtr<Stage>& stages,
                     TensorVarMap<TensorVarPtr>* vmap,
                     TensorVarMap<TensorVarPtr>* rvmap) {
  for (StagePtr s : stages->element) {
    OpPtr op = ReplaceInputs(s->op, *vmap);
    if (op != s->op) {
      for (size_t i = 0; i < op->output_count(); i++) {
        auto it = rvmap->find(s->op->output(i));
        if (it != rvmap->end()) {
          (*vmap)[it->second] = op->output(i);
        } else {
          (*vmap)[s->op->output(i)] = op->output(i);
          (*rvmap)[op->output(i)] = s->op->output(i);
        }
      }
      s->op = op;
    }
  }
}

TensorVarPtr Schedule::cache_read(const TensorVarPtr& tensor,
                                  const std::string& scope,
                                  const Array<Op>& readers) {
  InvalidateCache();
  std::ostringstream os;
  os << tensor->get_name() << "." << scope;
  TensorVarMap<TensorVarPtr> vsub;
  StagePtr s = operator[](tensor->op);
  TensorVarPtr stensor = s->op->output(0);
  // FIXME: what can I do here?
  // - [TensorVar]s are now transient ...
  // - ... so what point creating a tcache?
  // - ... and how should it behave?
  //  TensorVarPtr tcache = std::make_shared<TensorVar>(
  //      std::make_shared<std::string>(os.str()), stensor->shape, stensor->op,
  //      stensor->get_dtype());
  auto shape = stensor->shape->element;
  auto iters = api::construct_indices(shape);
  std::vector<ir::ExprPtr> iter_vector;
  for (auto& iter : iters.element) {
    iter_vector.push_back(iter);
  }
  auto tcache = api::compute(shape, iters, (*stensor)(iter_vector), os.str());

  vsub[stensor] = tcache;
  TensorVarMap<TensorVarPtr> vmap;
  TensorVarMap<TensorVarPtr> rvmap;
  for (OpPtr op : readers.element) {
    StagePtr s = operator[](op);
    OpPtr replace_op = ReplaceInputs(s->op, vsub);
    //    ELENA_ASSERT_NE(replace_op, s->op,
    //                    "Can't find given tensor in reader when doing
    //                    cache_read");
    vmap[s->op->output(0)] = replace_op->output(0);
    rvmap[replace_op->output(0)] = s->op->output(0);
    s->op = replace_op;
  }
  ReplaceDataFlow(stages, &vmap, &rvmap);
  StagePtr ostage = operator[](tensor->op);
  size_t pos = stages->findvar(ostage);
  StagePtr cstage = std::make_shared<Stage>(tcache->op);
  cstage->scope = scope;
  stages->element.insert(stages->element.begin() + pos + 1, cstage);
  op_stage->element[tcache->op] = cstage;
  cstage->group = ostage->group;
  if (cstage->group != nullptr) {
    ++cstage->group->num_child_stages;
  }
  return tcache;
}

void PrepareAxisMap(StagePtr ostage, ComputeOpPtr cop,
                    std::unordered_set<IterVarPtr>* p_reduce_axis,
                    ArrayPtr<IterVar> new_axis,
                    ir::MapPtr<IterVar, Range> dom_map,
                    std::unordered_map<NodePtr, NodePtr>* p_vsub,
                    std::unordered_map<VarPtr, ExprPtr>* p_vsub2newvar,
                    std::vector<Expr>* p_predicates) {
  auto& vsub = *p_vsub;
  ir::MapPtr<IterVar, Expr> value_map =
      std::make_shared<ir::Map<IterVar, Expr>>();
  for (auto iter : ostage->leaf_itervars->element) {
    if (iter->is_reduce) continue;
    RangePtr dom = dom_map->element.at(iter);
    IterVarPtr new_iv = std::make_shared<IterVar>(dom, iter->get_name() + ".c");
    new_axis->element.push_back(new_iv);
    value_map->element[iter] = new_iv;
  }
  // skip reduction iteration
  for (auto iv : cop->iter_vars->element) {
    if (iv->is_reduce) continue;
    vsub[iv] = value_map->element.at(iv);
  }
}

ArrayPtr<TensorVar> ReplaceOriginalOp(SchedulePtr self, StagePtr ostage,
                                      const std::string& scope, OpPtr cache_op,
                                      OpPtr origin_new_op, size_t tsize) {
  ArrayPtr<TensorVar> cache_tensor_list = std::make_shared<Array<TensorVar>>();
  for (size_t i = 0; i < tsize; i++) {
    cache_tensor_list->element.push_back(cache_op->output(i));
  }
  TensorVarMap<TensorVarPtr> vmap;
  TensorVarMap<TensorVarPtr> rvmap;
  for (size_t i = 0; i < tsize; i++) {
    vmap[ostage->op->output(i)] = origin_new_op->output(i);
    rvmap[origin_new_op->output(i)] = ostage->op->output(i);
  }
  ReplaceDataFlow(self->stages, &vmap, &rvmap);
  auto op_tmp = ostage->op;
  ostage->op = origin_new_op;
  ostage->all_itervars = std::make_shared<ir::Array<IterVar>>(
      ir::ptr_cast<ComputeOp>(ostage->op)->iter_vars->element);
  ostage->leaf_itervars =
      std::make_shared<ir::Array<IterVar>>(ostage->all_itervars->element);
  ostage->relations = std::make_shared<Array<IterVarRelation>>();

  size_t pos = self->stages->findvar(ostage);
  StagePtr cstage = std::make_shared<Stage>(cache_op);
  cstage->scope = scope;
  self->stages->element.insert(self->stages->element.begin() + pos, cstage);
  self->op_stage->element[cache_op] = cstage;

  cstage->group = ostage->group;
  if (cstage->group != nullptr) {
    ++cstage->group->num_child_stages;
  }
  for (size_t i = 0; i < self->outputs->element.size(); i++) {
    if (self->outputs->element[i] == op_tmp) {
      self->outputs->element[i] = origin_new_op;
    }
  }
  return cache_tensor_list;
}

ArrayPtr<TensorVar> CacheWriteWithReLayout(SchedulePtr self,
                                           const ArrayPtr<TensorVar>& tensorarr,
                                           const std::string& scope) {
  self->InvalidateCache();
  TensorVarPtr tensor = tensorarr->element[0];
  StagePtr ostage = (*self)[tensor->op];
  ComputeOpPtr cop = ir::ptr_cast<ComputeOp>(ostage->op);
  std::unordered_set<IterVarPtr> reduce_axis;
  ArrayPtr<IterVar> new_axis = std::make_shared<Array<IterVar>>();
  MapPtr<IterVar, Range> dom_map = std::make_shared<ir::Map<IterVar, Range>>();

  std::unordered_map<NodePtr, NodePtr> vsub;
  std::unordered_map<VarPtr, ExprPtr> vsub_to_newv;
  std::vector<Expr> predicates;

  // TODO(jimy): passDownDomain Step Need to be checked
  for (auto iter : cop->iter_vars->element) {
    dom_map->element[iter] = std::make_shared<ir::Range>(iter->range);
  }
  passDownDomain(ostage, dom_map);

  PrepareAxisMap(ostage, cop, &reduce_axis, new_axis, dom_map, &vsub,
                 &vsub_to_newv, &predicates);
  ExprPtr body;
  auto replacer = VarReplacer(vsub);
  body = ir::ptr_cast<Expr>(replacer.mutateReplace(cop->fcompute));
  // TODO(jimy): new_fcompute;   injectPredicate to do
  // TODO(jimy): new_fcompute;   VarReplacer.Mutate to do
  ir::MapPtr<IterVar, Expr> value_map =
      std::make_shared<ir::Map<IterVar, Expr>>();
  for (auto iter : cop->iter_vars->element) {
    value_map->element[iter] = iter;
  }
  // TODO(jimy): PassDownIndex  to do
  ir::ArrayPtr<ir::IterVar> args = std::make_shared<ir::Array<ir::IterVar>>();
  for (auto& iv : ostage->leaf_itervars->element) {
    if (iv->is_reduce) continue;
    args->element.push_back(ir::ptr_cast<IterVar>(value_map->element[iv]));
  }
  std::vector<ir::ExprPtr> args_vector;
  for (auto arg : args->element) {
    args_vector.push_back(arg);
  }

  OpPtr cache_op = ComputeOp::create(cop->output_shape(0), new_axis, body,
                                     cop->output(0)->get_name() + ".c");

  TensorVarPtr ctensor = cache_op->output(0);
  OpPtr origen_new_op =
      ComputeOp::create(cop->output_shape(0), args, (*ctensor)(args_vector),
                        cop->output(0)->get_name());
  // TODO(jimy): support multi output op
  size_t output_num = 1;
  return ReplaceOriginalOp(self, ostage, scope, cache_op, origen_new_op,
                           output_num);
}

ArrayPtr<TensorVar> Schedule::cache_write(const ArrayPtr<TensorVar>& tensor,
                                          const std::string& scope) {
  return CacheWriteWithReLayout(shared_from_this(), tensor, scope);
}

TensorVarPtr Schedule::cache_write(const TensorVarPtr& tensor,
                                   const std::string& scope) {
  return CacheWriteWithReLayout(
             shared_from_this(),
             std::make_shared<Array<TensorVar>>(
                 std::initializer_list<TensorVarPtr>{tensor}),
             scope)
      ->element[0];
}

StagePtr Copystage(const StagePtr& s) {
  StagePtr sc = std::make_shared<Stage>();
  ELENA_ASSERT_NE(s, nullptr, "Can't copy a null stage");
  *sc = *s;
  return sc;
}

SchedulePtr Schedule::copy_self() const {
  std::unordered_map<StagePtr, StagePtr> smap;
  SchedulePtr n = std::make_shared<Schedule>();
  n->outputs = outputs;

  for (StagePtr s : stages->element) {
    StagePtr sc = Copystage(s);
    smap[s] = sc;
    n->stages->element.push_back(sc);
  }
  for (StagePtr s : groups->element) {
    StagePtr sc = Copystage(s);
    smap[s] = sc;
    n->groups->element.push_back(sc);
  }
  for (auto k : op_stage->element) {
    n->op_stage->element[k.first] = smap.at(k.second);
  }
  for (StagePtr s : n->stages->element) {
    if (s->attach_stage->size() != 0) {
      for (size_t i = 0; i < s->attach_stage->size(); ++i) {
        ELENA_ASSERT_NE(smap.find(s->attach_stage->element[i]), smap.end(),
                        "Can't find the stage in stages attach stage in smap");
        s->attach_stage->element[i] = smap.at(s->attach_stage->element[i]);
      }
    }
    if (s->group != nullptr) {
      ELENA_ASSERT_NE(smap.find(s->group), smap.end(),
                      "Can't find the stage in stages group stage in smap");
      s->group = smap.at(s->group);
    }
  }
  for (StagePtr s : n->groups->element) {
    if (s->attach_stage->size() != 0) {
      for (size_t i = 0; i < s->attach_stage->size(); ++i) {
        ELENA_ASSERT_NE(smap.find(s->attach_stage->element[i]), smap.end(),
                        "Can't find the stage in stages attach stage in smap");
        s->attach_stage->element[i] = smap.at(s->attach_stage->element[i]);
      }
    }
    if (s->group != nullptr) {
      ELENA_ASSERT_NE(smap.find(s->group), smap.end(),
                      "Can't find the stage in groups group stage in smap");
      s->group = smap.at(s->group);
    }
  }
  return n;
}

void RebaseNonZeroMinLoop(const SchedulePtr& sch) {
  std::unordered_map<IterVarPtr, IterVarPtr> rebase_map;
  for (StagePtr s : sch->stages->element) {
    if (s->attach_type == AttachType::InlinedAlready) continue;

    OpPtr opp = s->op;
    ArrayPtr<IterVar> root_iters = std::make_shared<Array<IterVar>>();
    if (opp->get_type() == IRNodeType::ComputeOp) {
      ComputeOpPtr cop = ir::ptr_cast<ComputeOp>(opp);
      for (IterVarPtr i : cop->iter_vars->element) {
        root_iters->element.push_back(i);
      }
      if (cop->fcompute->get_type() == IRNodeType::Reduce) {
        ReducePtr rdc = ir::ptr_cast<Reduce>(cop->fcompute);
        for (IterVarPtr i : rdc->reduce_axis->element) {
          root_iters->element.push_back(i);
        }
      }
    }
    for (IterVarPtr i : root_iters->element) {
      size_t pos = s->leaf_itervars->findvar(i);
      auto it = s->iter_attr->element.find(i);
      if (it != s->iter_attr->element.end() &&
          (*it).second->bind_thread != nullptr) {
        continue;
      }
      if (pos < s->leaf_itervars->element.size()) {
        IterVarPtr rebased = std::make_shared<IterVar>(
            std::make_shared<Range>(), i->get_name(), i->is_reduce);
        s->relations->element.push_back(
            std::make_shared<RebaseRelation>(i, rebased));
        if (s->iter_attr->element.count(i)) {
          s->iter_attr->element[rebased] = s->iter_attr->element.at(i);
        }
        rebased->iter_type = i->iter_type;
        s->leaf_itervars->element[pos] = rebased;
        rebase_map[i] = rebased;
      }
    }
  }
  for (StagePtr s : sch->stages->element) {
    if (s->attach_type != AttachType::Scope) continue;
    for (size_t i = 0; i < s->attach_var->size(); ++i) {
      if (rebase_map.count(s->attach_var->element[i])) {
        s->attach_var->element[i] = rebase_map.at(s->attach_var->element[i]);
      }
    }
  }
  for (StagePtr s : sch->groups->element) {
    if (s->attach_type != AttachType::Scope) continue;
    for (size_t i = 0; i < s->attach_var->size(); ++i) {
      if (rebase_map.count(s->attach_var->element[i])) {
        s->attach_var->element[i] = rebase_map.at(s->attach_var->element[i]);
      }
    }
  }
}

void InjectInline(SchedulePtr sch) {
  sch->InvalidateCache();

  std::vector<ExprPtr> new_fcompute(sch->stages->element.size());
  std::vector<bool> is_changed(sch->stages->element.size(), false);

  for (int i = sch->stages->element.size(); i != 0; --i) {
    StagePtr stage = sch->stages->element[i - 1];
    if (stage->attach_type == AttachType::Inline) {
      stage->attach_type = AttachType::InlinedAlready;
      const ComputeOpPtr cop = ir::ptr_cast<ComputeOp>(stage->op);
      ELENA_ASSERT_EQ(stage->op->get_type(), IRNodeType::ComputeOp,
                      "Only compute op can reach here");
      ArrayPtr<IterVar> args = std::make_shared<Array<IterVar>>();
      ExprPtr fcompute = cop->fcompute;
      std::copy(cop->iter_vars->element.begin(), cop->iter_vars->element.end(),
                std::back_inserter(args->element));
      for (int j = i; j < sch->stages->element.size(); ++j) {
        StagePtr s = sch->stages->element[j];
        if (s->op->get_type() == IRNodeType::ComputeOp) {
          const ComputeOpPtr copj = ir::ptr_cast<ComputeOp>(s->op);
          new_fcompute[j] = copj->fcompute;
          ExprPtr new_value =
              inlineExpr(new_fcompute[j], stage->op, args, fcompute);

          copj->fcompute = new_value;
        }
      }
    }
  }
}
SchedulePtr Schedule::normalize() {
  SchedulePtr sch = copy_self();
  InjectInline(sch);
  // RebaseNonZeroMinLoop(sch);
  return sch;
}

}  // namespace ir
