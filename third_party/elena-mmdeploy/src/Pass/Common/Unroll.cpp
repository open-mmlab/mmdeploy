#include "Pass/Common/Unroll.h"

#include "IR/Expr.h"
#include "api.h"

ir::StmtPtr substituteExpr(
    ir::StmtPtr stmt, std::unordered_map<std::string, ir::ExprPtr> value_map) {
  if (value_map.size() == 0) return stmt;
  return ir::ptr_cast<ir::Stmt>(IRSubstitue(value_map).visit(stmt.get()));
}

void blockSplit(ir::StmtPtr stmt, std::vector<ir::StmtPtr> &blk_list) {
  if (!stmt) return;
  if (stmt->get_type() == ir::IRNodeType::Block) {
    auto block = ir::ptr_cast<ir::Block>(stmt);
    if (block->head) {
      blockSplit(block->head, blk_list);
    }
    if (block->tail) {
      blockSplit(block->tail, blk_list);
    }
  } else {
    blk_list.push_back(stmt);
  }
}

ir::NodePtr UnrollFuller::visit(ir::For *node) {
  mutate(node->body);
  auto iter = node->it;
  auto body = node->body;
  auto init = ir::ptr_cast<Const<uint64_t>>(node->init);
  auto extent = ir::ptr_cast<Const<uint64_t>>(node->extent);
  auto init_ = static_cast<size_t>(init->get_value());
  auto extent_ = static_cast<size_t>(extent->get_value());
  if (iter->iter_type == ir::IterAttrType::Unrolled) {
    std::unordered_map<std::string, ir::ExprPtr> value_map;
    std::vector<StmtPtr> blk_;
    for (size_t i = init_ + 1; i < extent_; i++) {
      value_map[iter->get_name()] =
          std::make_shared<Const<uint64_t>>(i, ir::ScalarType::UInt64);
      StmtPtr blk = ir::ptr_cast<ir::Stmt>(stmt_copyer.stmt_copy(body));
      blk = substituteExpr(blk, value_map);
      blk_.push_back(blk);
    }
    value_map[iter->get_name()] =
        std::make_shared<Const<uint64_t>>(0, ir::ScalarType::UInt64);
    body = substituteExpr(body, value_map);
    std::vector<ir::StmtPtr> blk_head;
    std::vector<ir::StmtPtr> blk_second;
    std::queue<StmtPtr> blockQueue;
    if (body->get_type() == ir::IRNodeType::Block) {
      blockQueue.push(body);
    }
    blockSplit(body, blk_head);
    for (size_t i = 0; i < blk_head.size(); i += offset) {
      for (size_t j = i + 1; j < i + offset; j++) {
        blk_head[i] = std::make_shared<ir::Block>(blk_head[i], blk_head[j]);
      }
    }
    for (auto stmt : blk_) {
      blk_second.clear();
      if (stmt->get_type() == ir::IRNodeType::Block) {
        blockQueue.push(stmt);
      }
      blockSplit(stmt, blk_second);
      for (size_t i = 0; i < blk_head.size(); i += offset) {
        for (size_t j = i + 1; j < i + offset; j++) {
          blk_second[i] =
              std::make_shared<ir::Block>(blk_second[i], blk_second[j]);
        }
        blk_head[i] = std::make_shared<ir::Block>(blk_head[i], blk_second[i]);
      }
    }
    body = nullptr;
    for (size_t i = 0; i < blk_head.size(); i += offset) {
      if (body) {
        body = std::make_shared<ir::Block>(body, blk_head[i]);
      } else {
        body = blk_head[i];
      }
    }
    offset *= extent_;
    return api::simplify(body);
  }
  return node->shared_from_this();
}

ir::NodePtr AutoUnroller::visit(ir::For *node) {
  mutate(node->body);
  if (is_unroll) {
    node->it->iter_type = ir::IterAttrType::Unrolled;
  }
  if (!is_full_unroll) is_unroll = false;
  return node->shared_from_this();
}

ir::NodePtr AutoUnroller::visit(ir::IfThenElse *node) {
  is_unroll = false;
  return node->shared_from_this();
}

ir::NodePtr AutoUnroller::visit(ir::Let *node) {
  is_unroll = false;
  return node->shared_from_this();
}

namespace api {

StmtPtr unrollFull(StmtPtr stmt) {
  UnrollFuller unroller;
  stmt = ir::ptr_cast<ir::Stmt>(unroller.visit(stmt.get()));
  return stmt;
}

StmtPtr autoUnroll(StmtPtr stmt, bool is_full_unroll) {
  AutoUnroller unroller(is_full_unroll);
  stmt = ir::ptr_cast<ir::Stmt>(unroller.visit(stmt.get()));
  return stmt;
}

}  // namespace api
