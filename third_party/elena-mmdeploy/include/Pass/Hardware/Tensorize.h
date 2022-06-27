#ifndef ELENA_INCLUDE_PASS_HARDWARE_TENSORIZE_H_
#define ELENA_INCLUDE_PASS_HARDWARE_TENSORIZE_H_

#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IR/Expr.h"
#include "IR/MutatorBase.h"
#include "IR/Type.h"
#include "api.h"

class Tensorizer : public MutatorBase<Tensorizer> {
 public:
  Tensorizer(int M_, int N_, int K_) : M(M_), N(N_), K(K_){};

  using MutatorBase::visit;
  ir::NodePtr visit(ir::For *node);
  ir::NodePtr visit(ir::Store *node);
  ir::NodePtr visit(ir::Attr *node);
  ir::NodePtr visit(ir::Allocate *node);

 private:
  int M, N, K;
  std::string mma_matrix_a;
  std::string mma_matrix_b;
  std::string mma_accumulator;
  std::vector<ir::Range> range;
  std::vector<ir::IterVarPtr> itervar;
  ExprPtr const_M =
      std::make_shared<ir::Const<uint64_t>>(M, ir::ScalarType::UInt64);
  ExprPtr const_N =
      std::make_shared<ir::Const<uint64_t>>(N, ir::ScalarType::UInt64);
  ExprPtr const_K =
      std::make_shared<ir::Const<uint64_t>>(K, ir::ScalarType::UInt64);
  ExprPtr const_MK =
      std::make_shared<ir::Const<uint64_t>>(M * K, ir::ScalarType::UInt64);
  ExprPtr const_MN =
      std::make_shared<ir::Const<uint64_t>>(M * N, ir::ScalarType::UInt64);
  ExprPtr const_NK =
      std::make_shared<ir::Const<uint64_t>>(N * K, ir::ScalarType::UInt64);
  int stack = 0;
};

namespace api {
ir::StmtPtr tensorize(ir::StmtPtr stmt, int M_ = 16, int N_ = 16, int K_ = 16);

}  // namespace api

#endif  // ELENA_INCLUDE_PASS_HARDWARE_TENSORIZE_H_
