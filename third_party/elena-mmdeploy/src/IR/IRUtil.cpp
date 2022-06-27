#include "IR/IRUtil.h"

#include <iostream>

#include "IR/VisitorBase.h"
#include "api.h"

using ir::Attr;
using ir::AttrPtr;
using ir::Binary;
using ir::Call;
using ir::For;
using ir::ForPtr;
using ir::IfThenElse;
using ir::IfThenElsePtr;
using ir::IRNodeType;
using ir::IterVar;
using ir::Let;
using ir::LetPtr;
using ir::ScalarVar;
using ir::StmtPtr;
using ir::TensorVar;

StmtPtr mergeNest(StmtPtr nest, StmtPtr body) {
  if (nest->get_type() == IRNodeType::For) {
    ForPtr for_stmt = ir::ptr_cast<For>(nest);
    return std::make_shared<For>(for_stmt->it, for_stmt->init, for_stmt->extent,
                                 body);
  } else if (nest->get_type() == IRNodeType::Let) {
    LetPtr let_stmt = ir::ptr_cast<Let>(nest);
    return std::make_shared<Let>(let_stmt->var, let_stmt->value, body);
  } else if (nest->get_type() == IRNodeType::IfThenElse) {
    IfThenElsePtr if_stmt = ir::ptr_cast<IfThenElse>(nest);
    return std::make_shared<IfThenElse>(if_stmt->condition, body, nullptr);
  } else if (nest->get_type() == IRNodeType::Attr) {
    AttrPtr attr_stmt = ir::ptr_cast<Attr>(nest);
    return std::make_shared<Attr>(attr_stmt->node, attr_stmt->key,
                                  attr_stmt->value, body);
  } else {
    abort();
  }
}

StmtPtr mergeNest(std::vector<StmtPtr> nest, StmtPtr body) {
  if (nest.size() == 0) {
    return body;
  }
  StmtPtr res;
  bool flag = true;
  for (int i = nest.size() - 1; i > -1; i--) {
    if (flag) {
      res = mergeNest(nest[i], body);
      flag = false;
    } else {
      res = mergeNest(nest[i], res);
    }
  }
  return res;
}

/// Cast Expr to instances of class IntSet.
class CastExpr2IntSet final : public VisitorBase<CastExpr2IntSet> {
 public:
  using VisitorBase<CastExpr2IntSet>::visit;

  /// Visit instances of class IterVar.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(iter_var);
  /// \encode
  ///
  /// \param iter_ptr instance of class IterVar;
  ///
  /// \return None.
  void visit(IterVar* iter_ptr) {
    // visit the arguments.
    mid_result = up_state[iter_ptr->shared_from_this()];
  }

  /// Visit instances of class Const.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(const);
  /// \encode
  ///
  /// \param const_ptr instance of class Const;
  ///
  /// \return None.
  void visit(Const<uint64_t>* const_ptr) {
    mid_result = std::make_shared<IntSet>();

    mid_result->setSinglePoint(const_ptr->shared_from_this());
  }

  /// Visit instances of class Const.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(const);
  /// \encode
  ///
  /// \param const_ptr instance of class Const;
  ///
  /// \return None.
  void visit(Binary* binary_ptr) {
    if (binary_ptr->operation_type == ir::BinaryType::Add) {
      visit(binary_ptr->lhs.get());
      IntSetPtr lhs = std::make_shared<IntSet>(*mid_result);
      visit(binary_ptr->rhs.get());
      IntSetPtr rhs = std::make_shared<IntSet>(*mid_result);
      mid_result = std::make_shared<IntSet>(*lhs + *rhs);
    } else if (binary_ptr->operation_type == ir::BinaryType::Mul) {
      visit(binary_ptr->lhs.get());
      IntSetPtr lhs = std::make_shared<IntSet>(*mid_result);
      visit(binary_ptr->rhs.get());
      IntSetPtr rhs = std::make_shared<IntSet>(*mid_result);
      printf("have not implement multiply for two IntSet\n");
      mid_result = std::make_shared<IntSet>(*lhs * *rhs);
    } else if (binary_ptr->operation_type == ir::BinaryType::Sub) {
      visit(binary_ptr->lhs.get());
      IntSetPtr lhs = std::make_shared<IntSet>(*mid_result);
      visit(binary_ptr->rhs.get());
      IntSetPtr rhs = std::make_shared<IntSet>(*mid_result);
      printf("have not implement multiply for two IntSet\n");
      mid_result = std::make_shared<IntSet>(*lhs - *rhs);
    } else if (binary_ptr->operation_type == ir::BinaryType::Div) {
      visit(binary_ptr->lhs.get());
      IntSetPtr lhs = std::make_shared<IntSet>(*mid_result);
      visit(binary_ptr->rhs.get());
      IntSetPtr rhs = std::make_shared<IntSet>(*mid_result);
      mid_result = std::make_shared<IntSet>(*lhs / *rhs);
    } else if (binary_ptr->operation_type == ir::BinaryType::Mod) {
      visit(binary_ptr->lhs.get());
      IntSetPtr lhs = std::make_shared<IntSet>(*mid_result);
      visit(binary_ptr->rhs.get());
      IntSetPtr rhs = std::make_shared<IntSet>(*mid_result);
      mid_result = std::make_shared<IntSet>(*lhs % *rhs);
    } else {
      std::string msg = "IntSet" + std::string(binary_ptr->get_type_name()) +
                        "not implemented";
      ELENA_ABORT(msg.c_str());
    }
  }

  /// Get the IntSet format of the given node.
  ///
  /// Typical Usage:
  /// \code
  ///   getIntSet(node, up_state);
  /// \encode
  ///
  /// \param up_state_ records the ranges of IterVars;
  ///
  /// \return the IntSet format of the node.
  IntSetPtr getIntSet(const ir::NodePtr& node, Rmap up_state_) {
    up_state = up_state_;
    mid_result = std::make_shared<IntSet>();
    visit(node.get());
    return mid_result;
  }

 private:
  IntSetPtr mid_result;
  Rmap up_state;
};

IntSetPtr expr2IntSet(ExprPtr expr, Rmap up_state_) {
  CastExpr2IntSet tmp_cast;
  return tmp_cast.getIntSet(expr, up_state_);
}

/// Get the range of root IterVars of the current stage.
class InputBound final : public VisitorBase<InputBound> {
 public:
  using VisitorBase::visit;

  /// Visit instances of class ScalarVar.
  ///
  /// Typical Usage:
  /// \code
  ///   visit(scalar);
  /// \encode
  ///
  /// \param scalar_ptr pointer to the instance of class ScalarVar;
  ///
  /// \return None.
  void visit(ScalarVar* scalar_ptr) {
    // visit the arguments.
    auto tensor_var = ir::ptr_cast<TensorVar>(scalar_ptr->tensor);
    if (tensor_var == nullptr) return;
    if (scalar_ptr->indices) visit(scalar_ptr->indices);
    if (tmap.count(tensor_var)) {
      for (int i = 0; i < scalar_ptr->indices->element.size(); i++) {
        tmap[tensor_var][i].push_back(
            expr2IntSet(scalar_ptr->indices->element[i], up_state));
      }
    }
  }

  /// Get the range of the root IterVars of the current stage.
  ///
  /// Typical Usage:
  /// \code
  ///   updateInputBound(expr, up_state, tmap);
  /// \encode
  ///
  /// \param expr usually the pointer to the expression containing
  /// the instance of class ScalarVar in terms of the output tensor
  /// of the current stage;
  ///
  /// \param up_state_ records the range of IterVars.
  ///
  /// \param tmap_ records the range of root IterVars of the
  /// output tensor of the current stage.
  ///
  /// \return None.
  void updateInputBound(
      ExprPtr expr, Rmap* up_state_,
      ir::TensorVarMap<std::vector<std::vector<IntSetPtr>>>* tmap_) {
    up_state = *up_state_;
    tmap = *tmap_;
    visit(expr.get());
    *up_state_ = up_state;
    *tmap_ = tmap;
  }

 private:
  Rmap up_state;
  ir::TensorVarMap<std::vector<std::vector<IntSetPtr>>> tmap;
};

void propBoundToInputs(
    ExprPtr expr, Rmap* up_state,
    ir::TensorVarMap<std::vector<std::vector<IntSetPtr>>>* tmap) {
  InputBound tmp;
  tmp.updateInputBound(expr, up_state, tmap);
}
