#include "Pass/Common/StatementSimplify.h"

#include "IR/Expr.h"
#include "api.h"

bool isConstType(ExprPtr node) {
  if (node->get_type() == IRNodeType::Const) {
    return true;
  }
  return false;
}

bool isConstNumber(ExprPtr node, uint64_t target) {
  if (isConstType(node)) {
    ir::ScalarType dtype = node->get_dtype();
    switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type: {                               \
    auto cb = static_cast<Const<native_type>*>(node.get());     \
    return cb->get_value() == (native_type)target;              \
  }
#include "x/scalar_types.def"
    }
  }
  return false;
}

NodePtr LoopSimplifier::visit(ir::For* node) {
  auto init = api::simplify(node->init);
  auto extent = api::simplify(node->extent);

  if (!isConstNumber(init, 0)) {
    if (!var_map.count(node->init)) {
      var_map[node->it] = node->init;
    } else {
      var_map[node->it] = var_map[node->init];
    }
    node->init = zero;
  }
  mutate(node->body);
  return node->shared_from_this();
}

NodePtr LoopSimplifier::visit(ir::IfThenElse* node) {
  if (!replace_set.count(node->condition)) {
    replace_set.insert(node->condition);
    mutate(node->condition);
  }
  mutate(node->then_case);
  if (node->else_case) mutate(node->else_case);
  return node->shared_from_this();
}

NodePtr LoopSimplifier::visit(ir::IterVar* node) {
  auto itervar_ptr = node->shared_from_this();

  auto it = var_map.find(itervar_ptr);
  if (it != var_map.end()) {
    return itervar_ptr + it->second;
  }

  auto iit = let_map.find(itervar_ptr);
  if (iit != let_map.end()) {
    return iit->second;
  }
  return itervar_ptr;
}

NodePtr LoopSimplifier::visit(ir::Let* node) {
  if (!let_map.count(node->value)) {
    let_map[node->var] = node->value;
  } else {
    let_map[node->var] = let_map[node->value];
  }
  mutate(node->value);
  mutate(node->body);
  return node->body;
}

uint64_t StatementSimplifier::computeMaxValue(ExprPtr x) {
  if (x->get_type() == IRNodeType::Const) {
    auto ptr = ir::ptr_cast<ir::Const<uint64_t>>(x);
    return ptr->get_value();
  }
  if (x->get_type() == IRNodeType::IterVar) {
    auto ptr = ir::ptr_cast<ir::IterVar>(x);
    auto it = range_map.find(ptr);
    if (it != range_map.end()) {
      return it->second - 1;
    } else {
      auto it = attr_map.find(ptr->get_name());
      if (it != attr_map.end()) {
        auto extent = std::make_shared<ir::Const<uint64_t>>(
            it->second, ir::ScalarType::UInt64);
        return computeMaxValue(ptr->range->init) +
               computeMaxValue(extent - one);
      } else {
        return computeMaxValue(ptr->range->init) +
               computeMaxValue(ptr->range->extent - one);
      }
    }
  }
  if (x->get_type() == IRNodeType::Binary) {
    auto ptr = ir::ptr_cast<ir::Binary>(x);
    if (ptr->operation_type == ir::BinaryType::Add) {
      return computeMaxValue(ptr->lhs) + computeMaxValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Mul) {
      return computeMaxValue(ptr->lhs) * computeMaxValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Sub) {
      return computeMaxValue(ptr->lhs) - computeMaxValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Div) {
      return computeMaxValue(ptr->lhs) / computeMaxValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Mod) {
      return computeMaxValue(ptr->lhs) % computeMaxValue(ptr->rhs);
    } else {
      ELENA_LOG_INFO("Too Conflict Binary");
      handle = false;
      return 1;
    }
  }
  if (x->get_type() == IRNodeType::Unary) {
    auto ptr = ir::ptr_cast<ir::Unary>(x);
    if (ptr->operation_type == ir::UnaryType::Ceil) {
      return ceil(computeMaxValue(ptr->operand));
    } else {
      ELENA_LOG_INFO("Too Conflict Unary");
      handle = false;
      return 1;
    }
  }
  ELENA_LOG_INFO("Too Conflict Expr");
  handle = false;
  return 1;
}

uint64_t StatementSimplifier::computeMinValue(ExprPtr x) {
  if (x->get_type() == IRNodeType::Const) {
    auto ptr = ir::ptr_cast<ir::Const<uint64_t>>(x);
    return ptr->get_value();
  }
  if (x->get_type() == IRNodeType::IterVar) {
    auto ptr = ir::ptr_cast<ir::IterVar>(x);
    return computeMinValue(ptr->range->init);
  }
  if (x->get_type() == IRNodeType::Binary) {
    auto ptr = ir::ptr_cast<ir::Binary>(x);
    if (ptr->operation_type == ir::BinaryType::Add) {
      return computeMinValue(ptr->lhs) + computeMinValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Mul) {
      return computeMinValue(ptr->lhs) * computeMinValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Sub) {
      return computeMinValue(ptr->lhs) - computeMinValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Div) {
      return computeMinValue(ptr->lhs) / computeMinValue(ptr->rhs);
    } else if (ptr->operation_type == ir::BinaryType::Mod) {
      return computeMinValue(ptr->lhs) % computeMinValue(ptr->rhs);
    } else {
      ELENA_LOG_INFO("Too Conflict Binary");
      handle = false;
      return 1;
    }
  }
  if (x->get_type() == IRNodeType::Unary) {
    auto ptr = ir::ptr_cast<ir::Unary>(x);
    if (ptr->operation_type == ir::UnaryType::Ceil) {
      return ceil(computeMinValue(ptr->operand));
    } else {
      ELENA_LOG_INFO("Too Conflict Unary");
      handle = false;
      return 1;
    }
  }
  ELENA_LOG_INFO("Too Conflict Expr");
  handle = false;
  return 1;
}

NodePtr StatementSimplifier::visit(ir::IfThenElse* node) {
  if (node->condition->get_type() != IRNodeType::Logical) {
    ELENA_LOG_INFO("Only Simplify If with LT Condition");
    handle = false;
    mutate(node->condition);
    mutate(node->then_case);
    if (node->else_case) mutate(node->else_case);
    return node->shared_from_this();
  }
  auto condition = ir::ptr_cast<ir::Logical>(node->condition);
  if (condition->operation_type == ir::LogicalType::LT) {
    uint64_t lhs = computeMaxValue(condition->lhs);
    uint64_t rhs = computeMinValue(condition->rhs);
    if (!handle) {
      handle = true;
      mutate(node->condition);
      mutate(node->then_case);
      if (node->else_case) mutate(node->else_case);
      return node->shared_from_this();
    } else if (lhs < rhs) {
      mutate(node->condition);
      mutate(node->then_case);
      if (node->else_case) mutate(node->else_case);
      return node->then_case;
    }
  }
  mutate(node->condition);
  mutate(node->then_case);
  if (node->else_case) mutate(node->else_case);
  return node->shared_from_this();
}

NodePtr StatementSimplifier::visit(ir::Let* node) {
  if (node->var->get_type() == ir::IRNodeType::IterVar) {
    auto ptr = ir::ptr_cast<ir::IterVar>(node->var);
    ptr->range->init = std::make_shared<Const<uint64_t>>(
        computeMinValue(node->value), ir::ScalarType::UInt64);
    ptr->range->extent =
        std::make_shared<Const<uint64_t>>(
            computeMaxValue(node->value) - computeMinValue(node->value),
            ir::ScalarType::UInt64) +
        one;
  }
  mutate(node->value);
  mutate(node->body);
  return node->shared_from_this();
}

NodePtr AttrSimplifier::visit(ir::Attr* node) {
  if (node->key == ir::AttrType::ThreadExtent) {
    ir::IterVarPtr var = ir::ptr_cast<ir::IterVar>(node->node);
    auto it = attr_map.find(var->get_name());
    if (it != attr_map.end()) {
      ir::ExprPtr var_extent = api::simplify(var->range->extent);
      if (var_extent->get_type() == ir::IRNodeType::Const) {
        auto ptr = static_cast<Const<uint64_t>*>(var_extent.get());
        if (ptr->get_value() < it->second) {
          var->range->extent = std::make_shared<Const<uint64_t>>(
              it->second, ir::ScalarType::UInt64);
        } else {
          attr_map[var->get_name()] = ptr->get_value();
        }
      }
    } else {
      ir::ExprPtr var_extent = api::simplify(var->range->extent);
      if (var_extent->get_type() == ir::IRNodeType::Const) {
        auto ptr = static_cast<Const<uint64_t>*>(var_extent.get());
        attr_map[var->get_name()] = ptr->get_value();
      }
    }
    return MutatorBase::visit(node);
  } else {
    return MutatorBase::visit(node);
  }
}

NodePtr StatementSimplifier::visit(ir::IterVar* node) {
  auto itervar_ptr = node->shared_from_this();

  auto it = var_map.find(itervar_ptr);
  if (it != var_map.end()) {
    return it->second;
  }
  return itervar_ptr;
}

NodePtr StatementSimplifier::visit(ir::For* node) {
  auto init = api::simplify(node->init);
  auto extent = api::simplify(node->extent);
  range_map[node->it] = computeMaxValue(init) + computeMaxValue(extent);
  if (isConstNumber(extent, 1)) {
    if (!var_map.count(node->init)) {
      var_map[node->it] = node->init;
    } else {
      var_map[node->it] = var_map[node->init];
    }
    mutate(node->it);
    mutate(node->init);
    mutate(node->extent);
    mutate(node->body);
    return node->body;
  }
  mutate(node->it);
  mutate(node->init);
  mutate(node->extent);
  mutate(node->body);
  return node->shared_from_this();
}

namespace api {

StmtPtr simplifyStatement(StmtPtr stmt) {
  AttrSimplifier attr_simplifier;
  stmt = ir::ptr_cast<ir::Stmt>(attr_simplifier.visit(stmt.get()));
  StatementSimplifier statement_simplifier(attr_simplifier.attr_map);
  stmt = ir::ptr_cast<ir::Stmt>(statement_simplifier.visit(stmt.get()));
  LoopSimplifier loop_simplifier;
  stmt = ir::ptr_cast<ir::Stmt>(loop_simplifier.visit(stmt.get()));
  return api::simplify(stmt);
}

}  // namespace api
