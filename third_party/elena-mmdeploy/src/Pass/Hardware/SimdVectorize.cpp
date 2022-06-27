//
// Created by SENSETIME\mupei on 2020/9/16.
//

#include "Pass/Hardware/SimdVectorize.h"

#include <utility>

#include "IR/Type.h"
#include "api.h"
#include "x/ir_node_types.def"

namespace ir {

// 1. Set symbol type for var (the left of equal sign) and value (the right of
// equal sign).
// 2. Substitute the current 'iter var' to 0.
// 3. Set lanes.
// PS: except for the 'Binary', 'Unary' and 'Logical' Node.
StmtPtr Vectorizer::visit(Store* op) {
  mutate(op->value);
  SymbolType value_symbol_type = Scalar_Symbol;
  if (op->value->get_type() == IRNodeType::ScalarVar) {
    ScalarVarPtr sv_value = ptr_cast<ScalarVar>(op->value);
    // remove iter and set symbol type
    if (sv_value->tensor && sv_value->indices) {
      for (int i = 0; i < sv_value->indices->size(); i++) {
        auto& ele = sv_value->indices->element[i];
        if (ele->get_type() == IRNodeType::IterVar && ele == loop_var_) {
          ele = std::make_shared<Const<uint64_t>>(0, ele->get_dtype());
          value_symbol_type = Vecotr_Symbol;
        }
      }
    }
  }
  mutate(op->var);
  bool need_vectorize_var = false;
  for (int i = 0; i < op->index->size(); i++) {
    auto& ele = op->index->element[i];
    if (ele->get_type() == IRNodeType::IterVar && ele == loop_var_) {
      ele = std::make_shared<Const<uint64_t>>(0, ele->get_dtype());
      need_vectorize_var = true;
    }
  }
  SymbolType var_symbol_type =
      need_vectorize_var ? Vecotr_Symbol : Broadcast_Symbol;
  // todo: Test if there's a bug in such condition:
  //  for (int i = 0; i < n1; i++) {
  //    for (int j = 0; j < n2; j++) {
  //      c[i * n2 +j] = a[i] + b[j];
  //    }
  //  }
  // which need to transform into:
  // VS(VS(c[0, 1, n1*n2])) =
  // VS(BS(a[0, 1, n2), 1, n1) + BS(VS(b[0, 1, n2], 1, n1))
  return std::make_shared<Store>(
      ptr_cast<Var>(setLanes(op->var, var_symbol_type)),
      setLanes(op->value, value_symbol_type), op->index);
}

StmtPtr Vectorizer::visit(IfThenElse* op) {
  // todo: concern when the condition is vector or broadcast
  return ptr_cast<Stmt>(MutatorBase::visit(op->then_case.get()));
}
StmtPtr Vectorizer::visit(Let* op) {
  return ptr_cast<Stmt>(MutatorBase::visit(op->body.get()));
}
StmtPtr Vectorizer::visit(Allocate* op) {
  return ptr_cast<Stmt>(MutatorBase::visit(op->body.get()));
}
ExprPtr Vectorizer::visit(Call* op) {
  return ptr_cast<Expr>(MutatorBase::visit(op));
}
ExprPtr Vectorizer::visit(Cast* op) {
  return ptr_cast<Expr>(MutatorBase::visit(op));
}
ExprPtr Vectorizer::visit(Unary* op) {
  return ptr_cast<Expr>(MutatorBase::visit(op));
}

// 1. Set symbol type for lhs and rhs.
// 2. Substitute the current 'iter var' to 0.
// 3. Set lanes by 'SetLanes' in 'VectorizeBinary'.
ExprPtr Vectorizer::visit(Binary* op) {
  VarPtr lhs_var = ptr_cast<Var>(op->lhs);
  VarPtr rhs_var = ptr_cast<Var>(op->rhs);
  auto modify_iter = [&](const VarPtr& var) {
    if (var->get_type() == IRNodeType::ScalarVar) {
      ScalarVarPtr sv_var = ptr_cast<ScalarVar>(var);
      if ((sv_var->tensor != nullptr) && (sv_var->indices != nullptr)) {
        size_t iter_index = sv_var->indices->findvar(loop_var_);
        if (iter_index != sv_var->indices->size()) {
          sv_var->indices->element[iter_index] =
              std::make_shared<Const<uint64_t>>(
                  0, sv_var->indices->element[iter_index]->get_dtype());
          return Vecotr_Symbol;
        } else {
          return Broadcast_Symbol;
        }
      }
    }
    ELENA_ABORT("Should never be here");
  };
  SymbolType lhs_symbol_type = modify_iter(lhs_var);
  SymbolType rhs_symbol_type = modify_iter(rhs_var);
  std::unordered_map<BinaryType, std::function<ExprPtr(ExprPtr, ExprPtr)>,
                     BinaryTypeHash>
      binary_func = {
          {BinaryType::Add,
           [](ExprPtr a, ExprPtr b) { return std::move(a) + std::move(b); }},
          {BinaryType::Sub,
           [](ExprPtr a, ExprPtr b) { return std::move(a) - std::move(b); }},
      };
  auto func_iter = binary_func.find(op->operation_type);
  if (func_iter != binary_func.end()) {
    return vectorizeBinary(op, func_iter->second, lhs_symbol_type,
                           rhs_symbol_type);
  } else {
    ELENA_ASSERT(false, "Do not support such binary operation currently!");
  }
}

ExprPtr Vectorizer::visit(Logical* op) {
  return ptr_cast<Expr>(MutatorBase::visit(op));
}

// Set lanes for Binary expr the return as vectorized binary expr.
template <typename T, typename FCompute>
ExprPtr Vectorizer::vectorizeBinary(const T* op, FCompute fcompute,
                                    SymbolType lhs_symbol_type,
                                    SymbolType rhs_symbol_type) {
  this->visit(op->lhs.get());
  this->visit(op->rhs.get());
  return fcompute(setLanes(op->lhs, lhs_symbol_type),
                  setLanes(op->rhs, rhs_symbol_type));
}

// Set lanes based on the previous lanes.
ExprPtr Vectorizer::setLanes(ExprPtr e, SymbolType symbol_type) const {
  uint64_t new_lanes = var_lanes_;
  VarPtr current_var = ptr_cast<Var>(e);
  std::stack<SymbolType> current_symbol_stack;
  if (e->get_type() == IRNodeType::VectorSymbol) {
    VectorSymbolPtr vs_ptr = ptr_cast<VectorSymbol>(e);
    new_lanes *= vs_ptr->get_lanes();
    current_symbol_stack = vs_ptr->get_current_symbol_stack();
  } else if (e->get_type() == IRNodeType::BroadcastSymbol) {
    BroadcastSymbolPtr bs_ptr = ptr_cast<BroadcastSymbol>(e);
    new_lanes *= bs_ptr->get_lanes();
    current_symbol_stack = bs_ptr->get_current_symbol_stack();
  }
  switch (symbol_type) {
    case Vecotr_Symbol:
      current_symbol_stack.emplace(Vecotr_Symbol);
      return std::make_shared<VectorSymbol>(e, current_symbol_stack, 1,
                                            new_lanes);
    case Broadcast_Symbol:
      current_symbol_stack.emplace(Broadcast_Symbol);
      return std::make_shared<BroadcastSymbol>(e, current_symbol_stack,
                                               new_lanes);
    case Scalar_Symbol:
      return e;
    default:
      ELENA_ASSERT(false, "No such symbol type!");
  }
}

// Decide whether vectorize or not.
StmtPtr LoopVectorizer::visit(For* op) {
  mutate(op->body);
  // todo: remove debug message
  op->for_type = ForType::Vectorized;
  if (op->for_type == ForType::Vectorized) {
    auto init_const_ptr = ptr_cast<Const<uint64_t>>(op->init);
    auto extent_const_ptr = ptr_cast<Const<uint64_t>>(op->extent);
    if (init_const_ptr == nullptr || extent_const_ptr == nullptr) {
      // todo: support dynamic shape
      ELENA_LOG_INFO("Do not support dynamic shape currently!");
      return ptr_cast<Stmt>(MutatorBase::visit(op->body.get()));
    }
    uint64_t init_value = init_const_ptr->get_value();
    uint64_t extent_value = extent_const_ptr->get_value();

    ELENA_ASSERT(init_value == 0, "The init value of the for op must be 0!");
    ELENA_ASSERT(
        extent_value >= 1,
        "The extent_const_ptr of the for op must larger than or equal to 1 "
        "when vectorization!");
    Vectorizer(op->it, extent_value).mutate(op->body);
    return op->body;
  } else {
    return std::make_shared<For>(op->it, op->init, op->extent, op->for_type,
                                 op->body);
  }
}

StmtPtr LoopVectorizerSkipper::visit(For* op) {
  ELENA_ASSERT(op != nullptr, "It's not a for stmt!");
  if (op->for_type == ForType::Vectorized) {
    return std::make_shared<For>(op->it, op->init, op->extent, op->for_type,
                                 op->body);
  } else {
    return ptr_cast<Stmt>(MutatorBase::visit(op->body.get()));
  }
}

}  // namespace ir
