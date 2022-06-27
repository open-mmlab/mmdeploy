#include "Pass/Hardware/Tensorize.h"

#include "IR/Expr.h"
#include "api.h"

ir::NodePtr Tensorizer::visit(ir::For *node) {
  range.insert(range.begin(), ir::Range(node->init, node->extent));
  itervar.insert(itervar.begin(), node->it);
  mutate(node->body);
  if (stack > 0) {
    stack -= 1;
    return node->body;
  }
  return node->shared_from_this();
}

ir::NodePtr Tensorizer::visit(ir::Store *node) {
  if (node->var->get_name() == mma_matrix_a ||
      node->var->get_name() == mma_matrix_b ||
      node->var->get_name() == mma_accumulator) {
    if (range.size() >= 2) {
      auto init_1 = ir::ptr_cast<ir::Const<uint64_t>>(range[0].init);
      auto extent_1 = ir::ptr_cast<ir::Const<uint64_t>>(range[0].extent);
      auto init_2 = ir::ptr_cast<ir::Const<uint64_t>>(range[1].init);
      auto extent_2 = ir::ptr_cast<ir::Const<uint64_t>>(range[1].extent);
      if (init_1->get_value() == 0 && init_2->get_value() == 0 &&
          extent_1->get_value() == K && extent_2->get_value() == M) {
        if (node->var->get_name() == mma_accumulator) {
          if (node->value->get_type() == ir::IRNodeType::Const) {
            std::vector<ir::ExprPtr> args;
            stack = 2;
            args.push_back(node->var);
            args.push_back(std::make_shared<ir::Const<uint64_t>>(
                M, ir::ScalarType::UInt64));
            args.push_back(std::make_shared<ir::Const<uint64_t>>(
                N, ir::ScalarType::UInt64));
            args.push_back(std::make_shared<ir::Const<uint64_t>>(
                K, ir::ScalarType::UInt64));
            args.push_back(api::simplify(
                (node->index->element[0] - itervar[0] - const_K * itervar[1]) /
                const_MK));
            args.push_back(std::make_shared<ir::Const<wchar_t>>(
                0, ir::ScalarType::Float16));
            return std::make_shared<ir::Evaluate>(std::make_shared<ir::Call>(
                ir::CallFunction::wmma_fill_fragment,
                std::make_shared<ir::Array<ir::Expr>>(args),
                ir::ScalarType::Float16));
          } else if (node->value->get_type() == ir::IRNodeType::Binary &&
                     range.size() >= 3) {
            auto init_3 = ir::ptr_cast<ir::Const<uint64_t>>(range[2].init);
            auto extent_3 = ir::ptr_cast<ir::Const<uint64_t>>(range[2].extent);
            if (init_3->get_value() == 0 && extent_3->get_value() == N) {
              auto binary_ptr = ir::ptr_cast<ir::Binary>(node->value);
              auto bin = binary_ptr->operation_type;
              auto lhs = binary_ptr->lhs;
              auto rhs = binary_ptr->rhs;
              if (rhs->get_type() == ir::IRNodeType::Binary) {
                auto rbinary_ptr = ir::ptr_cast<ir::Binary>(rhs);
                auto rbin = rbinary_ptr->operation_type;
                auto rlhs = rbinary_ptr->lhs;
                auto rrhs = rbinary_ptr->rhs;
                if (bin == ir::BinaryType::Add && rbin == ir::BinaryType::Mul &&
                    rlhs->get_type() == ir::IRNodeType::ScalarVar &&
                    rrhs->get_type() == ir::IRNodeType::ScalarVar) {
                  auto rl = ir::ptr_cast<ir::ScalarVar>(rlhs);
                  auto rr = ir::ptr_cast<ir::ScalarVar>(rrhs);
                  if (rl->tensor->get_name() == mma_matrix_a &&
                      rr->tensor->get_name() == mma_matrix_b) {
                    std::vector<ir::ExprPtr> args;
                    stack = 3;
                    args.push_back(node->var);
                    args.push_back(
                        api::simplify((node->index->element[0] - itervar[1] -
                                       const_K * itervar[2]) /
                                      const_MK));
                    args.push_back(rl->tensor);
                    args.push_back(
                        api::simplify((rl->indices->element[0] - itervar[0] -
                                       const_N * itervar[2]) /
                                      const_MN));
                    args.push_back(rr->tensor);
                    args.push_back(
                        api::simplify((rr->indices->element[0] -
                                       const_K * itervar[0] - itervar[1]) /
                                      const_NK));
                    args.push_back(node->var);
                    args.push_back(
                        api::simplify((node->index->element[0] - itervar[1] -
                                       const_K * itervar[2]) /
                                      const_MK));
                    return std::make_shared<ir::Evaluate>(
                        std::make_shared<ir::Call>(
                            ir::CallFunction::wmma_mma_sync,
                            std::make_shared<ir::Array<ir::Expr>>(args),
                            ir::ScalarType::Float32));
                  }
                }
              }
            }
          }
        } else if (node->value->get_type() == ir::IRNodeType::ScalarVar) {
          auto scalar_ptr = ir::ptr_cast<ir::ScalarVar>(node->value);
          std::vector<ir::ExprPtr> args;
          stack = 2;
          args.push_back(node->var);
          ExprPtr index;
          if (node->var->get_name() == mma_matrix_a) {
            index = api::simplify(
                (node->index->element[0] - itervar[0] - const_N * itervar[1]));
            args.push_back(index / const_MN);
          } else if (node->var->get_name() == mma_matrix_b) {
            index = api::simplify(
                (node->index->element[0] - itervar[0] - const_K * itervar[1]));
            args.push_back(index / const_NK);
          }
          args.push_back(scalar_ptr->tensor);
          args.push_back(api::simplify(scalar_ptr->indices->element[0] -
                                       node->index->element[0] + index));
          args.push_back(std::make_shared<ir::Const<uint64_t>>(
              16, ir::ScalarType::UInt64));
          return std::make_shared<ir::Evaluate>(std::make_shared<ir::Call>(
              ir::CallFunction::wmma_load_matrix_sync,
              std::make_shared<ir::Array<ir::Expr>>(args),
              ir::ScalarType::Float16));
        }
      }
    }
  } else if (node->value->get_type() == ir::IRNodeType::ScalarVar) {
    auto scalar_ptr = ir::ptr_cast<ir::ScalarVar>(node->value);
    if (scalar_ptr->tensor->get_name() == mma_accumulator) {
      std::vector<ir::ExprPtr> args;
      stack = 2;
      auto index = api::simplify((scalar_ptr->indices->element[0] - itervar[0] -
                                  const_K * itervar[1]));
      args.push_back(node->var);
      args.push_back(api::simplify(node->index->element[0] -
                                   scalar_ptr->indices->element[0] + index));
      args.push_back(scalar_ptr->tensor);
      args.push_back(index / const_MK);
      args.push_back(
          std::make_shared<ir::Const<uint64_t>>(16, ir::ScalarType::UInt64));
      return std::make_shared<ir::Evaluate>(std::make_shared<ir::Call>(
          ir::CallFunction::wmma_store_matrix_sync,
          std::make_shared<ir::Array<ir::Expr>>(args),
          ir::ScalarType::Float16));
    }
  }
  return node->shared_from_this();
}

ir::NodePtr Tensorizer::visit(ir::Allocate *node) {
  if (node->var->get_name() == mma_matrix_a) {
    node->is_tensorize = true;
    node->args.push_back(0);
    node->args.push_back(M);
    node->args.push_back(N);
    node->args.push_back(K);
    node->args.push_back(static_cast<int>(ir::ScalarType::Float16));
    node->bound->element[0]->extent =
        api::simplify(node->bound->element[0]->extent / const_MN);
  } else if (node->var->get_name() == mma_matrix_b) {
    node->is_tensorize = true;
    node->args.push_back(1);
    node->args.push_back(M);
    node->args.push_back(N);
    node->args.push_back(K);
    node->args.push_back(static_cast<int>(ir::ScalarType::Float16));
    node->bound->element[0]->extent =
        api::simplify(node->bound->element[0]->extent / const_NK);
  } else if (node->var->get_name() == mma_accumulator) {
    node->is_tensorize = true;
    node->args.push_back(2);
    node->args.push_back(M);
    node->args.push_back(N);
    node->args.push_back(K);
    node->args.push_back(static_cast<int>(ir::ScalarType::Float32));
    node->bound->element[0]->extent =
        api::simplify(node->bound->element[0]->extent / const_MK);
  }
  mutate(node->var);
  mutate(node->bound);
  mutate(node->body);
  return node->shared_from_this();
}

ir::NodePtr Tensorizer::visit(ir::Attr *node) {
  if (node->key == ir::AttrType::StorageScope) {
    auto label = ir::ptr_cast<ir::Label>(node->value);
    auto tensor = ir::ptr_cast<ir::TensorVar>(node->node);
    if (label->get_value() == "wmma.matrix_a") {
      mma_matrix_a = tensor->get_name();
    } else if (label->get_value() == "wmma.matrix_b") {
      mma_matrix_b = tensor->get_name();
    } else if (label->get_value() == "wmma.accumulator") {
      mma_accumulator = tensor->get_name();
    }
  }
  mutate(node->node);
  mutate(node->value);
  mutate(node->body);
  return node->shared_from_this();
}

namespace api {

StmtPtr tensorize(StmtPtr stmt, int M_, int N_, int K_) {
  Tensorizer tensorizePass(M_, N_, K_);
  stmt = ir::ptr_cast<ir::Stmt>(tensorizePass.visit(stmt.get()));
  return api::simplify(stmt);
}

}  // namespace api
