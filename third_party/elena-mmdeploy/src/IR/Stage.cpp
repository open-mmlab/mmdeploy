#include "IR/Stage.h"

#include <algorithm>
#include <fstream>

#include "IR/IRUtil.h"
#include "IR/VarReplacer.h"
#include "Pass/Common/StmtCopy.h"
#include "Schedule/TensorReplacer.h"
#include "api.h"
#include "logging.h"

namespace ir {

IterVarRelation::IterVarRelation(IRNodeType type) : Node(type) {}

SplitRelation::SplitRelation(IterVarPtr _parent, IterVarPtr _outer,
                             IterVarPtr _inner, ExprPtr _factor,
                             ExprPtr _nparts)
    : IterVarRelation(type),
      parent(_parent),
      outer(_outer),
      inner(_inner),
      factor(_factor),
      nparts(_nparts) {}

FuseRelation::FuseRelation(IterVarPtr _inner, IterVarPtr _outer,
                           IterVarPtr _fused)
    : IterVarRelation(type), inner(_inner), outer(_outer), fused(_fused) {}

SingletonRelation::SingletonRelation(IterVarPtr _iter)
    : IterVarRelation(type), iter(_iter) {}

RebaseRelation::RebaseRelation(IterVarPtr _parent, IterVarPtr _rebased)
    : IterVarRelation(type), parent(_parent), rebased(_rebased) {}

IterAttr::IterAttr() : Node(IRNodeType::IterAttr) {
  prefetch_data = std::make_shared<Array<TensorVar>>();
  prefetch_offset = std::make_shared<Array<Expr>>();
}

void IterAttr::set_bind_thread(IterVarPtr i) { bind_thread = i; }
void IterAttr::set_attr_type(IterAttrType k) { attr_type = k; }

Stage::Stage(OpPtr _op)
    : Node(IRNodeType::Stage),
      region_merge(_op->output_shape(0)->size(), true) {
  all_itervars = std::make_shared<Array<IterVar>>();
  leaf_itervars = std::make_shared<Array<IterVar>>();
  iter_attr = std::make_shared<Map<IterVar, IterAttr>>();
  relations = std::make_shared<Array<IterVarRelation>>();
  attach_stage = std::make_shared<Array<Stage>>();
  attach_var = std::make_shared<Array<IterVar>>();

  op = _op;
  origin_op = _op;
  if (_op->get_type() == IRNodeType::ComputeOp) {
    ComputeOpPtr cop = ir::ptr_cast<ComputeOp>(_op);
    std::copy(cop->iter_vars->element.begin(), cop->iter_vars->element.end(),
              std::back_inserter(all_itervars->element));
    std::copy(cop->iter_vars->element.begin(), cop->iter_vars->element.end(),
              std::back_inserter(leaf_itervars->element));
    if (cop->fcompute->get_type() == IRNodeType::Reduce) {
      ReducePtr rdc = ir::ptr_cast<Reduce>(cop->fcompute);
      for (IterVarPtr i : rdc->reduce_axis->element) {
        all_itervars->element.push_back(i);
        leaf_itervars->element.push_back(i);
      }
    }
  }
}

Stage::Stage() : Node(IRNodeType::Stage) {
  all_itervars = std::make_shared<Array<IterVar>>();
  leaf_itervars = std::make_shared<Array<IterVar>>();
  iter_attr = std::make_shared<Map<IterVar, IterAttr>>();
  relations = std::make_shared<Array<IterVarRelation>>();
  attach_stage = std::make_shared<Array<Stage>>();
  attach_var = std::make_shared<Array<IterVar>>();
}

void Split(StagePtr self, IterVarPtr parent, ExprPtr factor, ExprPtr nparts,
           IterVarPtr* outer, IterVarPtr* inner) {
  ArrayPtr<IterVar> all_iter = self->all_itervars;
  ArrayPtr<IterVar> leaf_iter = self->leaf_itervars;
  size_t pos = leaf_iter->findvar(parent);
  ELENA_ASSERT_NE(pos, leaf_iter->element.size(),
                  "Can't find given iteration in this stage when doing split");

  IterVarPtr p_outer = std::make_shared<IterVar>(std::make_shared<Range>(),
                                                 parent->get_name() + ".outer",
                                                 parent->is_reduce);
  IterVarPtr p_inner = std::make_shared<IterVar>(std::make_shared<Range>(),
                                                 parent->get_name() + ".inner",
                                                 parent->is_reduce);
  p_outer->iter_type = parent->iter_type;
  p_inner->iter_type = parent->iter_type;

  self->relations->element.push_back(std::make_shared<SplitRelation>(
      parent, p_outer, p_inner, factor, nparts));

  all_iter->element.push_back(p_outer);
  all_iter->element.push_back(p_inner);
  leaf_iter->element.erase(leaf_iter->element.begin() + pos);
  leaf_iter->element.insert(leaf_iter->element.begin() + pos, p_inner);
  leaf_iter->element.insert(leaf_iter->element.begin() + pos, p_outer);
  *outer = p_outer;
  *inner = p_inner;
}

StagePtr Stage::split(IterVarPtr parent, ExprPtr factor, IterVarPtr* p_outer,
                      IterVarPtr* p_inner) {
  Split(shared_from_this(), parent, factor, nullptr, p_outer, p_inner);
  return shared_from_this();
}

Array<IterVar> Stage::split(IterVarPtr parent, ExprPtr factor) {
  IterVarPtr outer, inner;
  split(parent, factor, &outer, &inner);
  return Array<IterVar>({outer, inner});
}

StagePtr Stage::split_nparts(IterVarPtr parent, ExprPtr nparts,
                             IterVarPtr* p_outer, IterVarPtr* p_inner) {
  Split(shared_from_this(), parent, nullptr, nparts, p_outer, p_inner);
  return shared_from_this();
}

Array<IterVar> Stage::split_nparts(IterVarPtr parent, ExprPtr nparts) {
  IterVarPtr outer, inner;
  split_nparts(parent, nparts, &outer, &inner);
  return Array<IterVar>({outer, inner});
}

StagePtr Stage::compute_at(StagePtr parent, IterVarPtr iter) {
  size_t pos = parent->leaf_itervars->findvar(iter);
  ELENA_ASSERT_NE(
      pos, parent->leaf_itervars->element.size(),
      "Can't find the iteration in parent's leaf_itervars when doing "
      "compute_at");
  if (attach_type != AttachType::Scope) {
    attach_type = AttachType::Scope;
    attach_stage->element.clear();
    attach_var->element.clear();
    attach_stage->element.push_back(parent);
    attach_var->element.push_back(iter);

  } else {
    size_t pos_stage = attach_stage->findvar(parent);
    if (pos_stage != attach_stage->element.size()) {
      attach_var->element[pos_stage] = iter;
    } else {
      attach_stage->element.push_back(parent);
      attach_var->element.push_back(iter);
    }
  }

  return shared_from_this();
}

StagePtr Stage::compute_inline() {
  attach_type = AttachType::Inline;
  return shared_from_this();
}

StagePtr Stage::compute_root() {
  attach_type = AttachType::GroupRoot;
  return shared_from_this();
}

StagePtr Stage::bind(IterVarPtr mvar, IterVarPtr tvar) {
  size_t pos = leaf_itervars->findvar(mvar);
  ELENA_ASSERT_NE(pos, leaf_itervars->element.size(),
                  "Can't find the iteration when doing bind");

  auto tmpiter = iter_attr->element.find(mvar);
  IterAttrPtr tmpp = std::make_shared<IterAttr>();
  if (tmpiter != iter_attr->element.end()) {
    tmpp = (*tmpiter).second;
  }
  tmpp->set_bind_thread(tvar);
  iter_attr->element[mvar] = tmpp;
  return shared_from_this();
}

StagePtr Stage::fuse(IterVarPtr outer, IterVarPtr inner, IterVarPtr* p_fused) {
  size_t pos_inner = leaf_itervars->findvar(inner);
  size_t pos_outer = leaf_itervars->findvar(outer);
  ELENA_ASSERT_NE(pos_inner, leaf_itervars->element.size(),
                  "Can't find inner iteration when doing fuse");
  ELENA_ASSERT_NE(pos_outer, leaf_itervars->element.size(),
                  "Can't find outer iteration when doing fuse");

  std::string fname = outer->get_name() + "." + inner->get_name() + ".fused";
  IterVarPtr fused =
      std::make_shared<IterVar>(std::make_shared<Range>(), fname);

  if (pos_inner + 1 == pos_outer) {
    std::swap(pos_inner, pos_outer);
    std::swap(inner, outer);
  }
  ELENA_ASSERT_EQ(pos_inner, pos_outer + 1,
                  "Can't fuse two iscontinuous iteration");

  relations->element.push_back(
      std::make_shared<FuseRelation>(inner, outer, fused));
  all_itervars->element.push_back(fused);
  leaf_itervars->element.erase(leaf_itervars->element.begin() + pos_outer,
                               leaf_itervars->element.begin() + pos_inner + 1);
  leaf_itervars->element.insert(leaf_itervars->element.begin() + pos_outer,
                                fused);
  *p_fused = fused;
  return shared_from_this();
}

StagePtr Stage::fuse(const ArrayPtr<IterVar>& axis, IterVarPtr* p_fused) {
  if (axis->element.size() != 0) {
    IterVarPtr fused = (*axis)[0];
    for (size_t i = 1; i < axis->element.size(); ++i) {
      fuse(fused, (*axis)[i], &fused);
    }
    *p_fused = fused;
  } else {
    IterVarPtr single = std::make_shared<IterVar>(
        std::make_shared<Const<uint64_t>>(0, ScalarType::UInt64),
        std::make_shared<Const<uint64_t>>(1, ScalarType::UInt64), "singleton");
    relations->element.push_back(std::make_shared<SingletonRelation>(single));
    all_itervars->element.push_back(single);
    leaf_itervars->element.insert(leaf_itervars->element.begin(), single);
    *p_fused = single;
  }
  return shared_from_this();
}

IterVarPtr Stage::fuse(std::initializer_list<IterVarPtr> t) {
  IterVarPtr fused;
  fuse(std::make_shared<Array<IterVar>>(t), &fused);
  return fused;
}

IterVarPtr Stage::fuse(std::vector<IterVarPtr> t) {
  IterVarPtr fused;
  fuse(std::make_shared<Array<IterVar>>(t), &fused);
  return fused;
}

StagePtr Stage::reorder(const ArrayPtr<IterVar>& order) {
  std::unordered_set<IterVarPtr> vis;
  for (auto i : order->element) {
    ELENA_ASSERT_EQ(vis.count(i), 0, "A iteration appears more than once");
    vis.insert(i);
  }

  std::vector<size_t> pos_order;
  for (size_t i = 0; i < order->element.size(); ++i) {
    pos_order.push_back(leaf_itervars->findvar((*order)[i]));
  }
  std::vector<IterVarPtr> tmp;
  for (size_t i = 0; i < pos_order.size(); ++i) {
    tmp.emplace_back((*leaf_itervars)[pos_order[i]]);
  }
  std::sort(pos_order.begin(), pos_order.end());
  for (size_t i = 0; i < pos_order.size(); ++i) {
    (*leaf_itervars)[pos_order[i]] = tmp[i];
  }
  return shared_from_this();
}
void Stage::reorder(std::initializer_list<IterVarPtr> t) {
  reorder(std::make_shared<Array<IterVar>>(t));
}

StagePtr Stage::tile(IterVarPtr x_parent, IterVarPtr y_parent, ExprPtr x_factor,
                     ExprPtr y_factor, IterVarPtr* x_outer, IterVarPtr* y_outer,
                     IterVarPtr* x_inner, IterVarPtr* y_inner) {
  split(x_parent, x_factor, x_outer, x_inner);
  split(y_parent, y_factor, y_outer, y_inner);
  reorder(std::make_shared<Array<IterVar>>(std::initializer_list<IterVarPtr>{
      *x_outer, *y_outer, *x_inner, *y_inner}));
  return shared_from_this();
}

Array<IterVar> Stage::tile(IterVarPtr x_parent, IterVarPtr y_parent,
                           ExprPtr x_factor, ExprPtr y_factor) {
  IterVarPtr x_outer, y_outer, x_inner, y_inner;
  tile(x_parent, y_parent, x_factor, y_factor, &x_outer, &y_outer, &x_inner,
       &y_inner);
  return Array<IterVar>(
      std::initializer_list<IterVarPtr>{x_outer, y_outer, x_inner, y_inner});
}

inline void Updateattr(StagePtr self, IterVarPtr iter, IterAttrType k) {
  size_t pos = self->leaf_itervars->findvar(iter);
  ELENA_ASSERT_NE(pos, self->leaf_itervars->element.size(),
                  "Can't find the iteration when update the iterattr");

  MapPtr<IterVar, IterAttr> iter_attr = self->iter_attr;
  auto it = iter_attr->element.find(iter);
  IterAttrPtr attr = std::make_shared<IterAttr>();
  if (it != iter_attr->element.end()) {
    attr = (*it).second;
  }
  attr->set_attr_type(k);
  iter_attr->element[iter] = attr;
  iter->iter_type = k;
}

StagePtr Stage::vectorize(IterVarPtr iter) {
  Updateattr(shared_from_this(), iter, IterAttrType::Vectorized);
  return shared_from_this();
}

StagePtr Stage::unroll(IterVarPtr iter) {
  Updateattr(shared_from_this(), iter, IterAttrType::Unrolled);
  return shared_from_this();
}

StagePtr Stage::parallel(IterVarPtr iter) {
  Updateattr(shared_from_this(), iter, IterAttrType::Parallelized);
  return shared_from_this();
}

StagePtr Stage::set_bind(IterVarPtr iter, std::string thread_tag) {
  iter->iter_type = IterAttrType::Thread;
  iter->thread_tag = thread_tag;
  iter->set_name(thread_tag);
  return shared_from_this();
}

StagePtr Stage::double_buffer() {
  ELENA_ASSERT(!is_output, "Can't use double_buffer to output");
  double_buffer_tag = true;
  return shared_from_this();
}

StagePtr Stage::set_align(IterVarPtr axis, int factor, int offset) {
  size_t pos = leaf_itervars->findvar(axis);
  ELENA_ASSERT_NE(pos, leaf_itervars->element.size(),
                  "Can't find the axis when set_align");

  auto it = iter_attr->element.find(axis);
  IterAttrPtr tmp;
  if (it != iter_attr->element.end()) {
    tmp = (*it).second;
  } else {
    tmp = std::make_shared<IterAttr>();
  }
  tmp->align_factor = factor;
  tmp->align_offset = offset;
  iter_attr->element[axis] = tmp;
  return shared_from_this();
}

StagePtr Stage::prefetch(const TensorVarPtr& tensor, IterVarPtr var,
                         ExprPtr offset) {
  size_t pos = leaf_itervars->findvar(var);
  ELENA_ASSERT_NE(pos, leaf_itervars->element.size(),
                  "Can't find the iteration when doing prefetch");

  auto it = iter_attr->element.find(var);
  IterAttrPtr tmp;
  if (it != iter_attr->element.end()) {
    tmp = (*it).second;
  } else {
    tmp = std::make_shared<IterAttr>();
  }
  tmp->prefetch_data->element.push_back(tensor);
  tmp->prefetch_offset->element.push_back(offset);
  iter_attr->element[var] = tmp;
  return shared_from_this();
}

StagePtr Stage::get_attach_spec() {
  auto spec = shared_from_this();
  while (spec->attach_type == AttachType::GroupRoot && spec->group) {
    spec = spec->group;
  }
  return spec;
}

void Stage::set_region_merge(std::vector<bool> b) {
  ELENA_ASSERT_EQ(b.size(), region_merge.size(),
                  "given bool vector size does not march region_merge size");
  region_merge = b;
}

void Stage::set_region_merge(int index, bool b) {
  ELENA_ASSERT(index < region_merge.size(),
               "index must be smaller than axis size when set_region_merge");
  region_merge[index] = b;
}

const std::vector<bool>& Stage::get_region_merge() const {
  return region_merge;
}

StagePtr Stage::set_scope(std::string s) {
  scope = s;
  return shared_from_this();
}

// Normal computation.
StmtPtr makeProvide(const ComputeOpPtr op) {
  ELENA_ASSERT_EQ(op->output_count(), 1,
                  "now only support op with only 1 output");
  ArrayPtr<Expr> index_array_ptr;
  if (op->tensor_indices != nullptr) {
    index_array_ptr = std::make_shared<Array<Expr>>();
    for (auto& iter : op->tensor_indices->element) {
      auto name = iter->get_name();
      if (iter->get_name().substr(0, 10) == "const_iter") {
        auto const_num = iter->range->init;
        assert(iter->range->init == iter->range->extent);
        index_array_ptr->element.push_back(const_num);
      } else {
        index_array_ptr->element.push_back(iter);
      }
    }
    // index_array_ptr = std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
    // op->tensor_indices->element.begin(), op->tensor_indices->element.end()});
  } else {
    index_array_ptr = std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
        op->iter_vars->element.begin(), op->iter_vars->element.end()});
  }

  auto res =
      std::make_shared<Provide>(op->output(0), op->fcompute, index_array_ptr);
  return res;
}

struct LoopNest {
  std::vector<StmtPtr> loop_list;
  std::vector<StmtPtr> bound_list;
  std::vector<StmtPtr> bound_list_init;
  std::vector<StmtPtr> let_list;
};

LoopNest getLoopNest(StagePtr stage, MapPtr<IterVar, Range> dom_map) {
  auto op = stage->op;
  ELENA_ASSERT_EQ(stage->op->get_type(), ir::IRNodeType::ComputeOp,
                  "op is not ComputeOp");
  auto compute_op = ir::ptr_cast<ComputeOp>(op);
  LoopNest result;
  for (auto loop_ : stage->leaf_itervars->element) {
    // assert this leaf itervars exist in dom_map
    if (!dom_map->element.count(loop_)) {
      dom_map->element[loop_] = std::make_shared<ir::Range>(loop_->range);
    }
    if (loop_->iter_type != IterAttrType::Thread) {
      result.loop_list.push_back(
          std::make_shared<For>(loop_, dom_map->element[loop_]->init,
                                dom_map->element[loop_]->extent, nullptr));
    } else if (loop_->get_name().length() >= 4 &&
               (loop_->get_name().substr(0, 4) == "iter" ||
                loop_->get_name().substr(0, 4) == "Redu")) {
      result.loop_list.push_back(
          std::make_shared<For>(loop_, dom_map->element[loop_]->init,
                                dom_map->element[loop_]->extent, nullptr));
    } else if (loop_->thread_tag.find("vthread") != loop_->thread_tag.npos) {
      result.loop_list.push_back(
          std::make_shared<Attr>(loop_, ir::AttrType::VirtualThread,
                                 dom_map->element[loop_]->extent, nullptr));
    } else {
      result.loop_list.push_back(
          std::make_shared<Attr>(loop_, ir::AttrType::ThreadExtent,
                                 dom_map->element[loop_]->extent, nullptr));

      // result.loop_list.push_back(
      //     std::make_shared<For>(loop_, dom_map->element[loop_]->init,
      //                           dom_map->element[loop_]->extent, nullptr));
    }
    if (loop_->range->init == nullptr) {
      // loop_->range->init = api::simplify(dom_map->element[loop_]->init);
      loop_->range->init = dom_map->element[loop_]->init;
    }
    if (loop_->range->extent == nullptr) {
      // loop_->range->extent = api::simplify(dom_map->element[loop_]->extent);
      loop_->range->extent = dom_map->element[loop_]->extent;
    }
  }
  for (const auto& rel : stage->relations->element) {
    // TODO(hanruobing): Add bound_list
    if (rel->get_type() == IRNodeType::SplitRelation) {
      auto r = ir::ptr_cast<SplitRelation>(rel);
      if (r->factor) {
        result.let_list.push_back(std::make_shared<Let>(
            ir::ptr_cast<Var>(r->parent),
            r->outer * r->factor + r->inner + dom_map->element[r->parent]->init,
            nullptr));
      } else {
        result.let_list.push_back(std::make_shared<Let>(
            ir::ptr_cast<Var>(r->parent),
            r->outer * std::make_shared<ir::Unary>(
                           dom_map->element[r->parent]->extent / r->nparts,
                           ir::UnaryType::Ceil) +
                r->inner + dom_map->element[r->parent]->init,
            nullptr));
      }
    } else if (rel->get_type() == IRNodeType::RebaseRelation) {
      auto r = ir::ptr_cast<RebaseRelation>(rel);
      if (r->parent->iter_type != ir::IterAttrType::Thread) {
        result.let_list.push_back(std::make_shared<Let>(
            ir::ptr_cast<Var>(r->parent),
            r->rebased + dom_map->element[r->parent]->init, nullptr));
      }
    } else if (rel->get_type() == IRNodeType::FuseRelation) {
      auto r = ir::ptr_cast<FuseRelation>(rel);
      result.let_list.push_back(
          std::make_shared<Let>(ir::ptr_cast<Var>(r->outer),
                                r->fused / dom_map->element[r->inner]->extent +
                                    dom_map->element[r->outer]->init,
                                nullptr));
      result.let_list.push_back(
          std::make_shared<Let>(ir::ptr_cast<Var>(r->inner),
                                r->fused % dom_map->element[r->inner]->extent +
                                    dom_map->element[r->inner]->init,
                                nullptr));
    } else {
      printf("not implement relation: %s for getLoopNest\n",
             rel->get_type_name());
      abort();
    }
  }
  // Add bound check
  if (stage->op->get_type() == ir::IRNodeType::ComputeOp) {
    auto compute_op = ir::ptr_cast<ir::ComputeOp>(stage->op);
    if (compute_op->fcompute->get_type() == IRNodeType::Reduce) {
      ReducePtr rdc = ir::ptr_cast<Reduce>(compute_op->fcompute);
      for (auto iv : rdc->reduce_axis->element) {
        result.bound_list.push_back(std::make_shared<IfThenElse>(
            std::make_shared<ir::Logical>(
                iv, iv->range->init + iv->range->extent, ir::LogicalType::LT),
            nullptr, nullptr));
        result.bound_list_init.push_back(std::make_shared<IfThenElse>(
            std::make_shared<ir::Logical>(
                iv, iv->range->init + iv->range->extent, ir::LogicalType::LT),
            nullptr, nullptr));
      }
    }
    for (auto iv : compute_op->iter_vars->element) {
      result.bound_list.push_back(std::make_shared<IfThenElse>(
          std::make_shared<ir::Logical>(iv, iv->range->init + iv->range->extent,
                                        ir::LogicalType::LT),
          nullptr, nullptr));
      result.bound_list_init.push_back(std::make_shared<IfThenElse>(
          std::make_shared<ir::Logical>(iv, iv->range->init + iv->range->extent,
                                        ir::LogicalType::LT),
          nullptr, nullptr));
    }
  }
  return result;
}

// Build a reduction body.
void makeReduction(const ComputeOpPtr op, ProvidePtr init, ProvidePtr provide) {
  ArrayPtr<Expr> args_init = std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
      op->iter_vars->element.begin(), op->iter_vars->element.end()});

  ArrayPtr<Expr> args_provide =
      std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
          op->iter_vars->element.begin(), op->iter_vars->element.end()});

  ELENA_ASSERT_EQ(op->fcompute->get_type(), ir::IRNodeType::Reduce,
                  "op\'s expr is not Reduce");
  const ReducePtr reduce = ir::ptr_cast<Reduce>(op->fcompute);

  // VarPtr lhs = reduce->accumulate;

  VarPtr lhs = op->output(0);

  ExprPtr init_value = reduce->init;

  // TODO(ruobing): replace accumulate in combiner to lhs
  ExprPtr update_value = reduce->combiner;

  *init = Provide(lhs, init_value, args_init);

  ArrayPtr<Expr> index = std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
      init->index->element.begin(), init->index->element.end()});

  std::unordered_map<ir::NodePtr, ir::NodePtr> vmap;
  // TODO(ruobing): what value should the string be?
  vmap[reduce->accumulate] =
      std::make_shared<ScalarVar>(init->var, index, "wtf");

  VarReplacer var_replace(vmap);
  update_value = ir::ptr_cast<Expr>(var_replace.mutateReplace(update_value));

  *provide = Provide(lhs, update_value, args_provide);
}

void makeCrossThreadReduction(const ComputeOpPtr op, ProvidePtr init,
                              EvaluatePtr atomic_part) {
  ArrayPtr<Expr> args_init = std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
      op->iter_vars->element.begin(), op->iter_vars->element.end()});

  ArrayPtr<Expr> args_provide =
      std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
          op->iter_vars->element.begin(), op->iter_vars->element.end()});

  ELENA_ASSERT_EQ(op->fcompute->get_type(), ir::IRNodeType::Reduce,
                  "op\'s expr is not Reduce");
  const ReducePtr reduce = ir::ptr_cast<Reduce>(op->fcompute);

  VarPtr lhs = op->output(0);

  ExprPtr init_value = reduce->init;

  ExprPtr update_value = reduce->combiner;

  *init = Provide(lhs, init_value, args_init);

  ArrayPtr<Expr> index = std::make_shared<Array<Expr>>(std::vector<ExprPtr>{
      init->index->element.begin(), init->index->element.end()});

  std::unordered_map<ir::NodePtr, ir::NodePtr> vmap;

  vmap[reduce->accumulate] =
      std::make_shared<ScalarVar>(init->var, index, "wtf");

  VarReplacer var_replace(vmap);
  update_value = ir::ptr_cast<Expr>(var_replace.mutateReplace(update_value));

  auto binary = ir::ptr_cast<Binary>(reduce->combiner);
  auto type = binary->operation_type;
  std::vector<ir::ExprPtr> atomic_args;
  atomic_args.push_back(binary->lhs);
  atomic_args.push_back(binary->rhs);
  CallPtr atomic_call_function;
  if (type == BinaryType::Add) {
    atomic_call_function = std::make_shared<ir::Call>(
        ir::CallFunction::atomic_add,
        std::make_shared<ir::Array<ir::Expr>>(atomic_args),
        binary->get_dtype());
  } else if (type == BinaryType::Max) {
    atomic_call_function = std::make_shared<ir::Call>(
                ir::CallFunction::atomic_max,
                std::make_shared<ir::Array<ir::Expr>>(atomic_args),
                binary->get_dtype());
  } else if (type == BinaryType::Min) {
    atomic_call_function = std::make_shared<ir::Call>(
                ir::CallFunction::atomic_min,
                std::make_shared<ir::Array<ir::Expr>>(atomic_args),
                binary->get_dtype());
  } else {
    ELENA_ABORT("Thread Level Reduce don\'t support this Binary Type.");
  }
  *atomic_part = Evaluate(atomic_call_function);
}

class CheckReduce : public VisitorBase<CheckReduce> {
 public:
  bool hasReduce(ir::StmtPtr body) {
    visit(body.get());
    return is_reduce;
  }

  bool hasReduce(ir::ExprPtr body) {
    visit(body.get());
    return is_reduce;
  }

  void visit(IterVar* iter) {
    if (iter->is_reduce) {
      is_reduce = true;
    }
  }

  using VisitorBase<CheckReduce>::visit;

 private:
  bool is_reduce = false;
};

StmtPtr makeInitPartStmt(LoopNest loop_nest, StmtPtr init) {
  std::vector<StmtPtr> no_reduce_axis;
  std::unordered_map<ir::NodePtr, ir::NodePtr> vmap;
  for (auto loop : loop_nest.loop_list) {
    auto for_loop = ir::ptr_cast<ir::For>(loop);
    if (for_loop) {
      auto axis = for_loop->it;
      if (!axis->is_reduce) {
        no_reduce_axis.push_back(loop);
        auto new_it = std::make_shared<ir::IterVar>(
            axis->range, axis->get_name() + "_init", axis->is_reduce);
        vmap[axis] = new_it;
      }
    } else {
      auto attr_loop = ir::ptr_cast<ir::Attr>(loop);
      if (attr_loop) {
        no_reduce_axis.push_back(loop);
      }
    }
  }
  std::vector<StmtPtr> no_reduce_bound;
  for (auto bound : loop_nest.bound_list_init) {
    if (!CheckReduce().hasReduce(bound)) {
      // init do not need bound check
      no_reduce_bound.push_back(bound);
    }
  }
  std::vector<StmtPtr> no_reduce_let;
  for (auto let : loop_nest.let_list) {
    if (!CheckReduce().hasReduce(let)) {
      no_reduce_let.push_back(let);
    }
  }
  init = mergeNest(no_reduce_bound, init);
  init = mergeNest(no_reduce_let, init);
  init = mergeNest(no_reduce_axis, init);
  VarReplacer var_replace(vmap);
  init = ir::ptr_cast<Stmt>(var_replace.mutateReplace(init));
  return init;
}

StmtPtr makeReducePartStmt(LoopNest outer_loop_nest,
                           const ir::ReducePtr reduce_ptr, StmtPtr provide) {
  std::vector<ir::StmtPtr> reduce_nest = outer_loop_nest.loop_list;
  provide = mergeNest(outer_loop_nest.bound_list, provide);
  provide = mergeNest(outer_loop_nest.let_list, provide);
  return mergeNest(outer_loop_nest.loop_list, provide);
}

StmtPtr mergeReducePair(StmtPtr init_part, StmtPtr reduce_part) {
  std::vector<StmtPtr> loop_nest;
  StmtPtr father = nullptr;
  auto init_iter = init_part;
  auto reduce_iter = reduce_part;

  // if init_part and reduce_part DO NOT have same outer most For loop, then
  // just block them
  if (init_iter->get_type() == IRNodeType::For &&
      reduce_iter->get_type() == IRNodeType::For) {
    auto iter_init = ir::ptr_cast<For>(init_iter)->it;
    auto iter_reduce = ir::ptr_cast<For>(reduce_iter)->it;
    if (iter_init != iter_reduce) {
      return std::make_shared<Block>(init_part, reduce_part);
    }
  }

  while (init_iter->get_type() == IRNodeType::For ||
         init_iter->get_type() == IRNodeType::Attr) {
    if (init_iter->get_type() == IRNodeType::For) {
      auto for_ptr_init = ir::ptr_cast<For>(init_iter);
      auto for_ptr_reduce = ir::ptr_cast<For>(reduce_iter);
      if (for_ptr_init->it != for_ptr_reduce->it) {
        break;
      }
      father = init_iter;
      init_iter = for_ptr_init->body;
      reduce_iter = for_ptr_reduce->body;
    }
    if (init_iter->get_type() == IRNodeType::Attr) {
      if (reduce_iter->get_type() == IRNodeType::For) {
        break;
      }
      auto attr_ptr_init = ir::ptr_cast<Attr>(init_iter);
      auto attr_ptr_reduce = ir::ptr_cast<Attr>(reduce_iter);
      if (attr_ptr_init->node != attr_ptr_reduce->node) {
        break;
      }
      father = init_iter;
      init_iter = attr_ptr_init->body;
      reduce_iter = attr_ptr_reduce->body;
    }
  }
  if (father->get_type() == IRNodeType::For) {
    auto for_ptr = ir::ptr_cast<For>(father);
    for_ptr->body = std::make_shared<Block>(init_iter, reduce_iter);
  } else if (father->get_type() == IRNodeType::Attr) {
    auto attr_ptr = ir::ptr_cast<Attr>(father);
    attr_ptr->body = std::make_shared<Block>(init_iter, reduce_iter);
  }
  return init_part;
}

StmtPtr makeComputeStmt(const StagePtr stage, MapPtr<IterVar, Range> dom_map) {
  ELENA_ASSERT_EQ(stage->op->get_type(), ir::IRNodeType::ComputeOp,
                  "only support ComputeOp");

  ComputeOpPtr op = ir::ptr_cast<ComputeOp>(stage->op);

  LoopNest loop_nest = getLoopNest(stage, dom_map);

  std::reverse(loop_nest.let_list.begin(), loop_nest.let_list.end());
  StmtPtr result;
  if (op->fcompute->get_type() == IRNodeType::Reduce) {
    if (ir::ptr_cast<Reduce>(op->fcompute)->cross_thread) {
      ProvidePtr init = std::make_shared<Provide>(nullptr, nullptr, nullptr);
      EvaluatePtr reduction = std::make_shared<Evaluate>(nullptr);
      makeCrossThreadReduction(op, init, reduction);
      stage->sync_type = 1;
      StmtPtr init_part_stmt = makeInitPartStmt(loop_nest, init);
      init_part_stmt = std::make_shared<ir::Block>(
          init_part_stmt,
          std::make_shared<ir::Evaluate>(std::make_shared<ir::Call>(
              ir::CallFunction::Sync, nullptr, ir::ScalarType::Boolean)));
      StmtPtr reduce_part_stmt =
          makeReducePartStmt(loop_nest, nullptr, reduction);
      result = std::make_shared<ir::Block>(init_part_stmt, reduce_part_stmt);
    } else {
      /*
      For Reduce Operation, now we will first generate two correspond stmt.
      For example, in the below case:
      for(i) {
        for(j) {
          a(i,j) = 0
          for(k) {
            a(i,j) += b(i,j,k)
          }
        }
      }
      we will generate below two stmt:
      First stmt is just for initilize.
      This Stmt will be generated by makeInitPartStmt()

      for(i) {
        for(j) {
          a(i,j) = 0
        }
      }

      The second stmt is only for accumulation
      This Stmt will be generated by makeReducePartStmt()

      for(i) {
        for(j) {
          for(k) {
            a(i,j) += b(i,j,k)
          }
        }
      }

      We will merge these stmt with common loop later
      */
      ProvidePtr init = std::make_shared<Provide>(nullptr, nullptr, nullptr);
      ProvidePtr reduction =
          std::make_shared<Provide>(nullptr, nullptr, nullptr);
      makeReduction(op, init, reduction);
      auto reduce = ir::ptr_cast<ir::Reduce>(op->fcompute);
      StmtPtr init_part_stmt = makeInitPartStmt(loop_nest, init);
      StmtPtr reduce_part_stmt =
          makeReducePartStmt(loop_nest, reduce, reduction);
      init_part_stmt = mergeReducePair(init_part_stmt, reduce_part_stmt);
      result = init_part_stmt;
    }
  } else {
    StmtPtr provide;
    if (op->fcompute->get_type() == ir::IRNodeType::Call)
      provide = std::make_shared<ir::Evaluate>(op->fcompute);
    else
      provide = makeProvide(op);
    provide = mergeNest(loop_nest.bound_list, provide);
    provide = mergeNest(loop_nest.let_list, provide);
    provide = mergeNest(loop_nest.loop_list, provide);
    result = provide;
  }
  // Add sync for provide
  if (stage->sync_type == 1) {
    // if (stage->scope.find("share") != stage->scope.npos) {
      result = std::make_shared<ir::Block>(
          result,
          std::make_shared<ir::Evaluate>(std::make_shared<ir::Call>(
              ir::CallFunction::Sync, nullptr, ir::ScalarType::Boolean)));
    // }
  }
  if (stage->sync_type == 2) {
    if (stage->scope.find("share") != stage->scope.npos) {
      result = std::make_shared<ir::Block>(
          std::make_shared<ir::Evaluate>(std::make_shared<ir::Call>(
              ir::CallFunction::Sync, nullptr, ir::ScalarType::Boolean)),
          result);
    }
  }
  if (op->attached_intrin_expr) {
    result = std::make_shared<ir::Block>(
        result, std::make_shared<Evaluate>(op->attached_intrin_expr));
  }
  return result;
}

StmtPtr makePipeline(StagePtr s, MapPtr<IterVar, Range> dom_map,
                     StmtPtr consumer) {
  // TODO(hanruobing): Add ProduceConsumer later
  StmtPtr producer = buildProvide(s, dom_map);
  StmtPtr pipeline = std::make_shared<Block>(producer, consumer);
  pipeline = buildRealize(s, dom_map, pipeline);

  // TODO(hanruobing) : add AttrStmt for storage_scope
  return pipeline;
}

// implement the provide utility.
StmtPtr buildProvide(StagePtr stage, MapPtr<IterVar, Range> dom_map) {
  /* TODO(hanruobing): support cross thread reduce and tensorize in future,
   * which is total different with normal Compute stmt
   */
  return makeComputeStmt(stage, dom_map);
}

// implement the provide utility.
StmtPtr buildRealize(StagePtr stage, MapPtr<IterVar, Range> dom_map,
                     StmtPtr body) {
  ELENA_ASSERT_EQ(stage->op->output_count(), 1,
                  "now only support op with only 1 output");
  ELENA_ASSERT_EQ(stage->op->get_type(), ir::IRNodeType::ComputeOp,
                  "only support ComputeOp");
  if (stage->op->get_type() == IRNodeType::ComputeOp) {
    auto op = ir::ptr_cast<ComputeOp>(stage->op);
    std::vector<RangePtr> bound;
    if (op->tensor_indices != nullptr) {
      std::vector<ir::ExprPtr> shape;
      for (auto& elem : op->output_shape(0)->element) {
        shape.push_back(elem);
      }
      ir::ExprPtr init = api::constant<uint64_t>(0);
      for (const auto& iv : shape) {
        ir::ExprPtr extent = iv;
        auto bound_ptr = std::make_shared<Range>(init, extent);
        bound.push_back(ir::ptr_cast<Range>(bound_ptr));
      }
    } else {
      for (const auto& iv : op->iter_vars->element) {
        bound.push_back(dom_map->element.at(iv));
        auto range = dom_map->element.at(iv);
      }
    }
    auto realize_stmt = std::make_shared<Realize>(
        op->output(0), std::make_shared<Array<Range>>(bound), body);
    // Author: XuPing
    if (stage->is_output) {
      realize_stmt->is_output = true;
    }
    if (!stage->scope.length()) {
      return realize_stmt;
    } else {
      return std::make_shared<Attr>(op->output(0), ir::AttrType::RealizeScope,
                                    std::make_shared<Label>(stage->scope),
                                    realize_stmt);
    }
  } else {
    abort();
  }
}

}  // namespace ir
