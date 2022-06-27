#include "Pass/Common/VectorizeLoad.h"

#include "IR/Expr.h"
#include "api.h"

bool is_const_(ExprPtr node) {
  if (node->get_type() == IRNodeType::Const) {
    return true;
  }
  return false;
}

bool get_const_number(ExprPtr node, int *target) {
  if (is_const_(node)) {
    ir::ScalarType dtype = node->get_dtype();
    switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type: {                               \
    auto cb = static_cast<Const<native_type> *>(node.get());    \
    *target = static_cast<int>(cb->get_value());                \
    return true;                                                \
  }
#include "x/scalar_types.def"
    }
  }
  return false;
}

ir::NodePtr Vectorizer::visit(ir::IfThenElse *node) {
  if_vectorize = false;
  mutate(node->then_case);
  if (node->else_case) {
    mutate(node->else_case);
  }
  return node->shared_from_this();
}

ir::NodePtr Vectorizer::visit(ir::Store *node) {
  auto index = node->index->element[0];

  mutate(node->index);
  if (if_vectorize) {
    node->index->element[0] = std::make_shared<ir::Ramp>(index, 1, 4);
  }
  mutate(node->value);
  return node->shared_from_this();
}

ir::NodePtr Vectorizer::visit(ir::ScalarVar *node) {
  if (!node->is_placeholder()) {
    auto indices = node->indices->element[0];
    mutate(node->indices);
    if (if_vectorize) {
      node->indices->element[0] = std::make_shared<ir::Ramp>(indices, 1, 4);
    }
  }
  return node->shared_from_this();
}

ir::NodePtr Vectorizer::visit(ir::IterVar *node) {
  if (node->get_name() == var_->get_name() && if_vectorize) {
    auto it = ir::ptr_cast<ir::IterVar>(
        stmt_copyer.stmt_copy(node->shared_from_this()));
    ExprPtr four =
        std::make_shared<ir::Const<uint64_t>>(4, ir::ScalarType::UInt64);
    return std::make_shared<ir::Binary>(it, four, ir::BinaryType::Mul);
  }
  return node->shared_from_this();
}

ir::NodePtr Vectorizer::visit(ir::Select *node) {
  mutate(node->cond);
  if (node->tBranch->get_type() == ir::IRNodeType::Const && if_vectorize) {
    node->tBranch = std::make_shared<ir::BroadcastSymbol>(node->tBranch, 4);
  } else {
    mutate(node->tBranch);
  }
  if (node->fBranch->get_type() == ir::IRNodeType::Const && if_vectorize) {
    node->fBranch = std::make_shared<ir::BroadcastSymbol>(node->fBranch, 4);
  } else {
    mutate(node->fBranch);
  }
  return node->shared_from_this();
}

ir::NodePtr Vectorizer::visit(ir::Binary *node) {
  if (node->operation_type == ir::BinaryType::Mul) {
    if (node->lhs->get_type() == ir::IRNodeType::IterVar &&
        node->rhs->get_type() == ir::IRNodeType::Const) {
      auto it = ir::ptr_cast<ir::IterVar>(node->lhs);
      auto num = ir::ptr_cast<ir::Const<uint64_t>>(node->rhs);
      if (it->get_name() == var_->get_name() && num->get_value() != 1) {
        if_vectorize = false;
      }
    } else if (node->rhs->get_type() == ir::IRNodeType::IterVar &&
               node->lhs->get_type() == ir::IRNodeType::Const) {
      auto it = ir::ptr_cast<ir::IterVar>(node->rhs);
      auto num = ir::ptr_cast<ir::Const<uint64_t>>(node->lhs);
      if (it->get_name() == var_->get_name() && num->get_value() != 1) {
        if_vectorize = false;
      }
    }
  }
  mutate(node->lhs);
  mutate(node->rhs);
  return node->shared_from_this();
}

ir::NodePtr VectorizeLoader::visit(ir::For *node) {
  auto iter = node->it;
  if (iter->iter_type == ir::IterAttrType::Vectorized) {
    int init = 0;
    int extent = 0;
    int lanes = 4;
    get_const_number(node->init, &init);
    get_const_number(node->extent, &extent);
    ExprPtr four =
        std::make_shared<ir::Const<uint64_t>>(4, ir::ScalarType::UInt64);
    if (init == 0 && extent % lanes == 0) {
      auto vectorizer_ = Vectorizer(iter, lanes);
      auto body = ir::ptr_cast<ir::Stmt>(vectorizer_.visit(node->body.get()));
      ExprPtr new_extent;
      if (vectorizer_.if_vectorize) {
        new_extent = api::simplify(std::make_shared<ir::Binary>(
            node->extent, four, ir::BinaryType::Div));
      } else {
        new_extent = node->extent;
      }
      return std::make_shared<ir::For>(node->it, node->init, new_extent, body);
    }
  }
  mutate(node->body);
  return node->shared_from_this();
}

namespace api {

StmtPtr vectorizeLoad(StmtPtr stmt) {
  VectorizeLoader vectorizeLoader;
  stmt = ir::ptr_cast<ir::Stmt>(vectorizeLoader.visit(stmt.get()));
  return stmt;
}

}  // namespace api
