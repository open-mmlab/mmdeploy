#include "IR/Bound.h"

#include "api.h"

/// Update the ranges of IterVars involved in split relations.
void passUpDomain(ir::SplitRelationPtr s,
                  ir::MapPtr<ir::IterVar, ir::Range> dom_map, IntSetPtr outer,
                  IntSetPtr inner, IntSetPtr parent) {
  /*
  TODO(hanruobing): add match_range
  if (dom_map->element.count(s->outer) && dom_map->element.count(s->inner) &&
  dom_map->element.count(s->parent) &&
  outer.match_range(dom_map.at(s->outer)) &&
  inner.match_range(dom_map.at(s->inner))) {
  *parent = IntSet::range(dom_map.at(s->parent));
  return;
  }
  */
  ir::ExprPtr factor = dom_map->element.at(s->inner)->extent;
  ir::ExprPtr parent_min = dom_map->element.at(s->parent)->init;
  *parent = IntSet(*outer * factor + *inner + parent_min);
}

/// Update the ranges of IterVars involved in fuse relations.
void passUpDomain(ir::FuseRelationPtr s,
                  ir::MapPtr<ir::IterVar, ir::Range> dom_map, IntSetPtr fused,
                  IntSetPtr outer, IntSetPtr inner) {
  ir::ExprPtr outer_min = dom_map->element.at(s->outer)->init;
  ir::ExprPtr inner_min = dom_map->element.at(s->inner)->init;
  ir::ExprPtr outer_extent = dom_map->element.at(s->outer)->extent;
  ir::ExprPtr inner_extent = dom_map->element.at(s->inner)->extent;
  // *outer = IntSet(ceil(*fused / inner_extent) + outer_min);
  // *inner = IntSet(*fused % inner_extent + inner_min);

  // outer->setRange(dom_map->element.at(s->outer));
  // inner->setRange(dom_map->element.at(s->inner));

  if (fused->isSinglePoint()) {
    ExprPtr outer_init =
        std::make_shared<ir::Binary>(fused->getRange()->init, inner_extent,
                                     ir::BinaryType::Div) +
        outer_min;
    outer_init = std::make_shared<ir::Unary>(outer_init, ir::UnaryType::Floor);
    outer->setSinglePoint(outer_init);

    ExprPtr inner_init =
        std::make_shared<ir::Binary>(fused->getRange()->init, inner_extent,
                                     ir::BinaryType::Mod) +
        inner_min;
    inner->setSinglePoint(inner_init);
  } else {
    outer->setRange(dom_map->element.at(s->outer));
    inner->setRange(dom_map->element.at(s->inner));
  }
}

/// Update the ranges of IterVars involved in fuse relations.
void passUpDomain(ir::FuseRelationPtr s, umap_iv_intset* up_state,
                  IntSetPtr fused, IntSetPtr outer, IntSetPtr inner,
                  ir::SplitRelationPtr r) {
  outer->setRange((*up_state).at(r->outer)->getRange());
  inner->setRange((*up_state).at(r->inner)->getRange());
}

/// Update the ranges of IterVars involved in rebase relations.
void passUpDomain(ir::RebaseRelationPtr s,
                  ir::MapPtr<ir::IterVar, ir::Range> dom_map, IntSetPtr rebased,
                  IntSetPtr parent) {
  ir::ExprPtr parent_min = dom_map->element.at(s->parent)->init;
  *parent = IntSet(*rebased + parent_min);
}

/// Invoking functions to update the ranges of IterVars
/// involved in various relations from leaves to root.
void passUpDomain(const ir::StagePtr stage, umap_iv_intset* up_state,
                  ir::MapPtr<ir::IterVar, ir::Range> dom_map) {
  for (size_t i = stage->relations->element.size(); i != 0; --i) {
    auto rel = stage->relations->element[i - 1];
    if (rel->get_type() == ir::IRNodeType::SplitRelation) {
      auto r = ir::ptr_cast<ir::SplitRelation>(rel);
      IntSetPtr parent = std::make_shared<IntSet>();

      passUpDomain(r, dom_map, (*up_state).at(r->outer),
                   (*up_state).at(r->inner), parent);
      (*up_state)[r->parent] = parent;
    } else if (rel->get_type() == ir::IRNodeType::RebaseRelation) {
      auto r = ir::ptr_cast<ir::RebaseRelation>(rel);
      IntSetPtr parent = std::make_shared<IntSet>();
      passUpDomain(r, dom_map, (*up_state).at(r->rebased), parent);
      (*up_state)[r->parent] = parent;
    } else if (rel->get_type() == ir::IRNodeType::FuseRelation) {
      auto r = ir::ptr_cast<ir::FuseRelation>(rel);
      IntSetPtr inner = std::make_shared<IntSet>();
      IntSetPtr outer = std::make_shared<IntSet>();
      passUpDomain(r, dom_map, (*up_state).at(r->fused), outer, inner);
      (*up_state)[r->outer] = outer;
      (*up_state)[r->inner] = inner;
    } else if (rel->get_type() == ir::IRNodeType::SingletonRelation) {
      // Do nothing, because the singleton is always one single point.
    } else {
      ELENA_WARN("not implement relationship< "
                 << rel->get_type_name() << " >for passUpDomain" << std::endl);
    }
  }
}

/// Update the range of IterVars from root to leaves
/// involved in various relations.
void passDownDomain(const ir::StagePtr stage,
                    ir::MapPtr<ir::IterVar, ir::Range> dom_map) {
  auto zero = api::constant<uint64_t>(0);
  auto one = api::constant<uint64_t>(1);
  for (const auto& rel : stage->relations->element) {
    if (rel->get_type() == ir::IRNodeType::SplitRelation) {
      auto r = ir::ptr_cast<ir::SplitRelation>(rel);
      if (!dom_map->element.count(r->parent)) {
        dom_map->element[r->parent] =
            std::make_shared<ir::Range>(r->parent->range);
      }
      auto parent_range = dom_map->element.at(r->parent);
      if (r->factor) {
        dom_map->element[r->inner] =
            std::make_shared<ir::Range>(zero, r->factor, one);
        dom_map->element[r->outer] = std::make_shared<ir::Range>(
            zero, ir::ceil_div(parent_range->extent, r->factor), one);
      } else {
        dom_map->element[r->inner] = std::make_shared<ir::Range>(
            zero, ir::ceil_div(parent_range->extent, r->nparts), one);
        dom_map->element[r->outer] =
            std::make_shared<ir::Range>(zero, r->nparts, one);
      }
    } else if (rel->get_type() == ir::IRNodeType::RebaseRelation) {
      auto r = ir::ptr_cast<ir::RebaseRelation>(rel);
      dom_map->element[r->rebased] = std::make_shared<ir::Range>(
          zero, dom_map->element.at(r->parent)->extent, one);
    } else if (rel->get_type() == ir::IRNodeType::FuseRelation) {
      auto r = ir::ptr_cast<ir::FuseRelation>(rel);
      if (!dom_map->element.count(r->inner)) {
        dom_map->element[r->inner] =
            std::make_shared<ir::Range>(r->inner->range);
      }
      if (!dom_map->element.count(r->outer)) {
        dom_map->element[r->outer] =
            std::make_shared<ir::Range>(r->outer->range);
      }
      dom_map->element[r->fused] =
          std::make_shared<ir::Range>(zero,
                                      dom_map->element[r->inner]->extent *
                                          dom_map->element[r->outer]->extent,
                                      one);
    } else if (rel->get_type() == ir::IRNodeType::SingletonRelation) {
      auto r = ir::ptr_cast<ir::SingletonRelation>(rel);
      dom_map->element[r->iter] = std::make_shared<ir::Range>(zero, one, one);
    } else {
      ELENA_ABORT("not implement relation for passDownDomain\n");
    }
  }
  return;
}

/// Record the range of IterVars when the stage corresponds
/// to the output tensor.
void inferOutputStage(const ir::StagePtr& stage,
                      const ir::MapPtr<ir::IterVar, ir::Range>& rmap) {
  auto op = ir::ptr_cast<ir::ComputeOp>(stage->op);
  if (op->tensor_indices != nullptr) {
    std::vector<ir::ExprPtr> shape;
    for (auto& elem : op->output_shape(0)->element) {
      shape.push_back(elem);
    }
    auto iters_shape =
        std::make_shared<ir::Array<ir::IterVar>>(api::construct_indices(shape));
    for (auto iv : iters_shape->element) {
      rmap->element[iv] = std::make_shared<ir::Range>(iv->range);
    }
    for (auto iv : op->iter_vars->element) {
      rmap->element[iv] = std::make_shared<ir::Range>(iv->range);
    }

  } else {
    for (auto iv : op->iter_vars->element) {
      rmap->element[iv] = std::make_shared<ir::Range>(iv->range);
    }
  }
  if (op->fcompute->get_type() == ir::IRNodeType::Reduce) {
    auto reduce_ptr = ir::ptr_cast<ir::Reduce>(op->fcompute);
    for (auto iv : reduce_ptr->reduce_axis->element) {
      rmap->element[iv] = std::make_shared<ir::Range>(iv->range);
    }
  }
}

/// Update the range of leaf IterVars when compute at schedule is applied.
void updateRootBound(const ir::StagePtr stage, const ir::IterVarPtr iv,
                     const bool found_attach,
                     const ir::MapPtr<ir::IterVar, ir::Range> rmap,
                     umap_iv_intset& up_state) {
  if (!found_attach ||
      (stage->scope == "__shared__" &&
       iv->thread_tag.find("threadIdx") != iv->thread_tag.npos)) {
    up_state[iv] = std::make_shared<IntSet>(rmap->element[iv]);
  } else {
    auto intset = std::make_shared<IntSet>();
    intset->setSinglePoint(iv);
    up_state[iv] = intset;
  }
}

/// Update the range of IterVars for all the stages.
void inferRootBound(
    const ir::StagePtr stage, ir::TensorVarMap<ir::ArrayPtr<ir::Op>> feed_graph,
    std::unordered_map<ir::OpPtr, ir::StagePtr> op2stage,
    std::unordered_map<ir::OpPtr, std::vector<ir::IterVarPtr>> attach_path,
    ir::MapPtr<ir::IterVar, ir::Range> rmap) {
  ELENA_ASSERT_NE(stage->attach_type, ir::AttachType::Inline,
                  "call schedule.normalize before scheduleops");
  if (stage->attach_type == ir::AttachType::InlinedAlready) return;

  // to avoid placeholder as direct output
  if (stage->op->get_type() == ir::IRNodeType::PlaceholderOp) {
    for (auto iv : stage->leaf_itervars->element) {
      rmap->element[iv] = std::make_shared<ir::Range>(iv->range);
    }
    return;
  }

  if (stage->is_output) {
    inferOutputStage(stage, rmap);
    return;
  }

  // The consumers of the op.
  ir::TensorVarMap<std::vector<std::vector<IntSetPtr>>> tmap;
  std::unordered_set<ir::OpPtr> consumers;
  for (int i = 0; i < stage->op->output_count(); i++) {
    auto t = stage->op->output(i);
    auto it = feed_graph.find(t);
    tmap.emplace(t, std::vector<std::vector<IntSetPtr>>(t->shape->size()));
    if (it != feed_graph.end()) {
      for (const auto& op : it->second->element) {
        consumers.insert(op);
      }
    }
  }

  if (consumers.size() == 0) {
    inferOutputStage(stage, rmap);
    return;
  }

  bool is_compute_at = (stage->attach_type == ir::AttachType::Scope);
  // TODO(hanruobing): support compuate_at to multi consumers
  ir::IterVarPtr attach_iter =
      is_compute_at ? stage->attach_var->element[0] : nullptr;

  ir::StagePtr compute_at_stage =
      is_compute_at ? stage->attach_stage->element[0] : nullptr;

  for (const auto& op : consumers) {
    umap_iv_intset up_state;
    bool found_attach = false;
    ir::StagePtr op_stage = op2stage.at(op);

    if (!(op_stage == compute_at_stage ||
          (op_stage->attach_stage->size() > 0 &&
           op_stage->attach_stage->element[0] == compute_at_stage)))
      continue;

    // Consumer nest
    for (size_t i = op_stage->leaf_itervars->size(); i != 0; --i) {
      ir::IterVarPtr iv = op_stage->leaf_itervars->element[i - 1];
      found_attach = (is_compute_at && (iv == attach_iter)) || found_attach;
      updateRootBound(stage, iv, found_attach, rmap, up_state);
    }

    for (const auto& iv : attach_path[op]) {
      found_attach = (is_compute_at && (iv == attach_iter)) || found_attach;
      updateRootBound(stage, iv, found_attach, rmap, up_state);
    }

    passUpDomain(op_stage, &up_state, rmap);

    evalAllItervarRange(up_state);

    // TODO(hanruobing): Relax if needed.
    if (op->get_type() == ir::IRNodeType::ComputeOp) {
      auto compute_op = ir::ptr_cast<ir::ComputeOp>(op);
      propBoundToInputs(compute_op->fcompute, &up_state, &tmap);
    }
  }
  ELENA_ASSERT_EQ(stage->op->output_count(), 1,
                  "Elena can only handle one output for a expr");
  auto compute_op = ir::ptr_cast<ir::ComputeOp>(stage->op);
  auto IntSet_vector = tmap.at(compute_op->output(0));
  auto output_tensor = compute_op->output(0);
  for (int i = 0; i < compute_op->iter_vars->element.size(); i++) {
    auto intset_list = tmap[output_tensor][i];
    if (intset_list.empty()) {
      // use origin range if true (now especially for norm,
      // because it needs ptr
      // in cuda)
      rmap->element[compute_op->iter_vars->element[i]] =
          std::make_shared<ir::Range>(compute_op->iter_vars->element[i]->range);
    } else {
      auto merge_intset = merge(&intset_list);
      rmap->element[compute_op->iter_vars->element[i]] =
          merge_intset.getRange();
    }
  }
  if (compute_op->fcompute->get_type() == ir::IRNodeType::Reduce) {
    auto reduce_ptr = ir::ptr_cast<ir::Reduce>(compute_op->fcompute);
    for (const auto& iv : reduce_ptr->reduce_axis->element) {
      rmap->element[iv] = std::make_shared<ir::Range>(iv->range);
    }
  }
}
