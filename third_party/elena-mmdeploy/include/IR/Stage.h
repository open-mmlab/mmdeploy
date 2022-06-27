#ifndef ELENA_INCLUDE_IR_STAGE_H_
#define ELENA_INCLUDE_IR_STAGE_H_

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "IR/Container.h"
#include "IR/Expr.h"
#include "IR/Op.h"
#include "Stmt.h"

namespace ir {
/**
 * @brief This is the base class of Schedule
 * @author lichuandong
 */
class IterVarRelation : public Node {
 public:
  /**
   * @brief constructor with irnodetype
   * @author lichuandong
   */
  explicit IterVarRelation(IRNodeType type);
  static const IRNodeType type = IRNodeType::IterVarRelation;
};
using IterVarRelationPtr = std::shared_ptr<IterVarRelation>;

/**
 * @brief parent has split relation with inner and outer
 * @author lichuandong
 */
class SplitRelation : public IterVarRelation,
                      public std::enable_shared_from_this<SplitRelation> {
 public:
  /**
   * @brief constructor for split relation
   * @author lichuandong
   */
  SplitRelation(IterVarPtr _parent, IterVarPtr _outer, IterVarPtr _inner,
                ExprPtr _factor, ExprPtr _nparts);

  static const IRNodeType type = IRNodeType::SplitRelation;
  IterVarPtr parent;
  IterVarPtr outer;
  IterVarPtr inner;
  ExprPtr factor;
  ExprPtr nparts;
};
using SplitRelationPtr = std::shared_ptr<SplitRelation>;

/**
 * @brief fuse relation between inner, outer and fused
 * @author: lichuandong
 */
class FuseRelation : public IterVarRelation,
                     public std::enable_shared_from_this<FuseRelation> {
 public:
  FuseRelation(IterVarPtr _inner, IterVarPtr _outer, IterVarPtr _fused);

  static const IRNodeType type = IRNodeType::FuseRelation;
  IterVarPtr inner;
  IterVarPtr outer;
  IterVarPtr fused;
};
using FuseRelationPtr = std::shared_ptr<FuseRelation>;

/**
 * @brief the singleton relation
 * @author lichuandong
 */
class SingletonRelation
    : public IterVarRelation,
      public std::enable_shared_from_this<SingletonRelation> {
 public:
  explicit SingletonRelation(IterVarPtr _iter);

  static const IRNodeType type = IRNodeType::SingletonRelation;
  IterVarPtr iter;
};
using SingletonRelationPtr = std::shared_ptr<SingletonRelation>;

/**
 * @brief the rebased relation
 * @author lichuandong
 */
class RebaseRelation : public IterVarRelation,
                       public std::enable_shared_from_this<RebaseRelation> {
 public:
  static const IRNodeType type = IRNodeType::RebaseRelation;
  explicit RebaseRelation(IterVarPtr _parent, IterVarPtr rebased);
  IterVarPtr parent;
  IterVarPtr rebased;
};
using RebaseRelationPtr = std::shared_ptr<RebaseRelation>;

class IterAttr : public Node, public std::enable_shared_from_this<IterAttr> {
 public:
  static const IRNodeType type = IRNodeType::IterAttr;
  IterAttr();
  void set_bind_thread(IterVarPtr i);
  void set_attr_type(IterAttrType k);

  IterVarPtr bind_thread;
  IterAttrType attr_type{IterAttrType::Data};
  ArrayPtr<TensorVar> prefetch_data;
  ArrayPtr<Expr> prefetch_offset;
  int align_offset{0};
  int align_factor{0};
};
using IterAttrPtr = std::shared_ptr<IterAttr>;

/**
 * @brief This is the base class of Schedule
 * @author lichuandong
 */
class Stage : public Node, public std::enable_shared_from_this<Stage> {
  using StagePtr = std::shared_ptr<Stage>;

 public:
  static const IRNodeType type = IRNodeType::Stage;
  /**
   * @brief constructor for compute opration
   * @author lichuandong
   */
  explicit Stage(OpPtr _op);
  Stage();
  /**
   * @brief split the iteration to inner and outer
   * @author lichuandong
   * @param parent the parent itervar
   * @param factor the split factor
   */
  StagePtr split(IterVarPtr parent, ExprPtr factor, IterVarPtr* p_outer,
                 IterVarPtr* p_inner);
  /**
   * @brief package the split operation to return an Array contains inner and
   * outer iteration
   * @author lichuandong
   * @param parent the parent itervar
   * @param factor the split factor
   */
  Array<IterVar> split(IterVarPtr parent, ExprPtr factor);
  /**
   * @brief split the iteration to inner and outer by given parts
   * @author lichuandong
   * @param parent the parent itervar
   * @param nparts the parts number
   */
  StagePtr split_nparts(IterVarPtr parent, ExprPtr nparts, IterVarPtr* p_outer,
                        IterVarPtr* p_inner);
  /**
   * @brief package the split_nparts operation to return an Array contains inner
   * and outer iteration
   * @author lichuandong
   * @param parent the parent itervar
   * @param nparts the parts number
   */
  Array<IterVar> split_nparts(IterVarPtr parent, ExprPtr nparts);
  /**
   * @brief compute the iter at the parent stage
   * @author lichuandong
   * @param parent parent satge
   * @param iter the iteration
   */
  StagePtr compute_at(StagePtr parent, IterVarPtr iter);
  /**
   * @brief compute inline
   * @author lichuandong
   */
  StagePtr compute_inline();
  /**
   * @brief compute at group root
   * @author lichuandong
   */
  StagePtr compute_root();
  /**
   * @brief bind a itervar to a thread
   * @author lichuandong
   * @param mvar the binded var
   * @param tvar the thread var
   */
  StagePtr bind(IterVarPtr mvar, IterVarPtr tvar);
  /**
   * @brief construct the environments for the threads to be launched
   * @author lichuandong
   * @param threads the threads to be launched
   */
  StagePtr environment_threads(ArrayPtr<IterVar> threads);
  /**
   * @brief fuse the inner and outer to fused
   * @author lichuandong
   * @param outer the outer var
   * @param inner the inner var
   * @param fused the fuse target
   */
  StagePtr fuse(IterVarPtr outer, IterVarPtr inner, IterVarPtr* fused);
  /**
   * @brief fuse all axis to fused
   * @author lichuandong
   * @param axis the outer var
   * @param fused the fuse target
   */
  StagePtr fuse(const ArrayPtr<IterVar>& axis, IterVarPtr* fused);
  /**
   * @brief package fuse operation to return fused iteration and accept an
   * Iteration initializer_list
   * @author lichuandong
   * @param t a continuous Iteration initializer_list
   */
  IterVarPtr fuse(std::initializer_list<IterVarPtr> t);
  IterVarPtr fuse(std::vector<IterVarPtr> t);
  /**
   * @brief reorder the leaf_itervars by order
   * @author lichuandong
   * @param order the new order
   */
  StagePtr reorder(const ArrayPtr<IterVar>& order);
  /**
   * @brief package reorder operation to accept an Iteration initializer_list
   * @author lichuandong
   * @param t an Iteration initializer_list
   */
  void reorder(std::initializer_list<IterVarPtr> t);
  /**
   * @brief tile x,y axis in the (x_outer,y_outer,x_inner,y_inner) order
   * @author lichuandong
   * @param x_parent the x parent axis
   * @param y_parent the y parent axis
   * @param x_factor the x split factor
   * @param y_factor the y split factor
   * @param x_outer the x outer axis
   * @param y_outer the y outer axis
   * @param x_inner the x inner axis
   * @param y_inner the y inner axis
   */
  StagePtr tile(IterVarPtr x_parent, IterVarPtr y_parent, ExprPtr x_factor,
                ExprPtr y_factor, IterVarPtr* x_outer, IterVarPtr* y_outer,
                IterVarPtr* x_inner, IterVarPtr* y_inner);
  /**
   * @brief package tile operation to return a iteration array
   * @author lichuandong
   * @param x_parent the x parent axis
   * @param y_parent the y parent axis
   * @param x_factor the x split factor
   * @param y_factor the y split factor
   * @return an iteration array with x_outer, y_outer, x_inner, y_inner order
   */
  Array<IterVar> tile(IterVarPtr x_parent, IterVarPtr y_parent,
                      ExprPtr x_factor, ExprPtr y_factor);
  /**
   * @brief vectorize a iteration
   * @author lichuandong
   * @param iter the iteration to be vectorized
   */
  StagePtr vectorize(IterVarPtr iter);
  /**
   * @brief unroll a iteration
   * @author lichuandong
   * @param iter the iteration to be unrolled
   */
  StagePtr unroll(IterVarPtr iter);
  /**
   * @brief parallel a iteration
   * @author lichuandong
   * @param iter the iteration to be paralleled
   */
  StagePtr parallel(IterVarPtr iter);

  /*
   * @brief set bind (block/thread) for the given iter
   * @author hanruobing
   * @param iter the iteration to be set
   * @param thread_tag a string, maybe blockIdx.y or threadIdx.x...
   */
  StagePtr set_bind(IterVarPtr iter, std::string thread_tag);

  /*
   * @brief compute this stage by double buffer
   * @author lichuandong
   * @param iter the iteration to be paralleled
   */
  StagePtr double_buffer();
  /**
   * @brief fetch data in advance
   * @author lichuandong
   * @param tensor the tensor to be prefetch
   * @param var the prefetching itervar
   * @param offset the prefetching itervar offset
   */
  StagePtr prefetch(const TensorVarPtr& tensor, IterVarPtr var, ExprPtr offset);
  /**
   * @brief set special alignment
   * @author lichuandong
   * @param axis the axis to be set
   * @param factor the alignment factor
   * @param offset the alignment offset
   */
  StagePtr set_align(IterVarPtr axis, int factor, int offset);
  /**
   * @brief Get attachment spec of current stage.
   * @author xupengcheng
   * @return stage that represents the attach spec of the current stage.
   */
  StagePtr get_attach_spec();

  /**
   * @brief set region_merge
   * @author lichuandong
   */
  void set_region_merge(std::vector<bool> b);
  /**
   * @brief set region_merge
   * @author lichuandong
   */
  void set_region_merge(int index, bool b);
  /**
   * @brief return this stage's op is region split or not
   * @author lichuandong
   */
  const std::vector<bool>& get_region_merge() const;

  /**
   * @brief modify scope to the given scope
   * @param scope memory scope
   * @author lichuandong
   */
  std::shared_ptr<Stage> set_scope(std::string scope);

  OpPtr op;
  ArrayPtr<IterVar> all_itervars;
  ArrayPtr<IterVar> leaf_itervars;
  ArrayPtr<IterVarRelation> relations;

  ArrayPtr<Stage> attach_stage;
  ArrayPtr<IterVar> attach_var;
  AttachType attach_type{AttachType::GroupRoot};

  MapPtr<IterVar, IterAttr> iter_attr;
  bool double_buffer_tag{false};

  OpPtr origin_op;
  StagePtr group;
  std::string scope;
  int num_child_stages{0};
  bool is_output{false};

  // show this stage's op need merge or split to other ops
  // true: merge, false: split
  std::vector<bool> region_merge;

  // mark the sync position
  // sync ... all shared scope... sync
  // 0: no sync 1: last sync 2: first sync
  int sync_type{0};
};
using StagePtr = std::shared_ptr<Stage>;

/**
 * @brief Create Provide Stmt for a stage
 * @param stage correspond stage
 * @param dom_map iter's range in the stage
 * @return The constructed Provide Stmt.
 */
StmtPtr buildProvide(StagePtr stage, MapPtr<IterVar, Range> dom_map);
/**
 * @brief Create Realize Stmt for this Op
 * @param stage correspond stage
 * @param dom_map iter's range in the stage
 * @return The constructed Provide Stmt.
 */
StmtPtr buildRealize(StagePtr stage, MapPtr<IterVar, Range> dom_map,
                     StmtPtr body);

StmtPtr makePipeline(StagePtr s, MapPtr<IterVar, Range> dom_map,
                     StmtPtr consumer);

}  // namespace ir

#endif  // ELENA_INCLUDE_IR_STAGE_H_
