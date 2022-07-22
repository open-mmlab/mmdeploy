#ifndef ELENA_INCLUDE_PASS_COMMON_STATEMENTSIMPLIFY_H_
#define ELENA_INCLUDE_PASS_COMMON_STATEMENTSIMPLIFY_H_

#include <cmath>
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

class StatementSimplifier : public MutatorBase<StatementSimplifier> {
 public:
  explicit StatementSimplifier(
      std::unordered_map<std::string, uint64_t> attr_map_)
      : attr_map(attr_map_) {}
  using MutatorBase::visit;
  ir::NodePtr visit(ir::IfThenElse* node);
  ir::NodePtr visit(ir::Let* node);
  ir::NodePtr visit(ir::IterVar* node);
  ir::NodePtr visit(ir::For* node);

 private:
  bool handle = true;
  uint64_t computeMaxValue(ExprPtr x);
  uint64_t computeMinValue(ExprPtr x);
  ExprPtr one =
      std::make_shared<ir::Const<uint64_t>>(1, ir::ScalarType::UInt64);
  std::unordered_map<ExprPtr, ExprPtr> var_map;
  std::unordered_map<ir::IterVarPtr, uint64_t> range_map;
  std::unordered_map<std::string, uint64_t> attr_map;
};

class LoopSimplifier : public MutatorBase<LoopSimplifier> {
 public:
  static ir::NodePtr LoopSimplify(ir::Node* node);

  using MutatorBase::visit;
  ir::NodePtr visit(ir::For* node);
  ir::NodePtr visit(ir::IterVar* node);
  ir::NodePtr visit(ir::Let* node);
  ir::NodePtr visit(ir::IfThenElse* node);

 private:
  std::unordered_map<ExprPtr, ExprPtr> var_map;
  std::unordered_map<ExprPtr, ExprPtr> let_map;
  std::unordered_set<ExprPtr> replace_set;
  ExprPtr zero =
      std::make_shared<ir::Const<uint64_t>>(0, ir::ScalarType::UInt64);
};

class AttrSimplifier : public MutatorBase<AttrSimplifier> {
 public:
  using MutatorBase::visit;
  ir::NodePtr visit(ir::Attr* node);
  std::unordered_map<std::string, uint64_t> attr_map;
};

namespace api {
/**
 * @brief Simplify For, IfThenElse and Let Statement
 * @author xuping
 * @param stmt the root node ptr to be Simplified
 * case1: Remove the For stmt with extent=1
 * Input:
 *  for (i = a + b; i < a + b + 1; i ++) {
 *    for (j = c; j < c + 2; j ++) {
 *      A[i - (a + b) + j] = 0;
 *    }
 *  }
 * Output:
 *  for (j = c; j < c + 2; j ++) {
 *    A[j] = 0;
 *  }
 * case2: Remove the IfThenElse stmt, if the contidion is always true
 * Input:
 *  for (i = 0; i < 1024; i ++) {
 *    if (i < 1024) {
 *      A[i] = 0;
 *    }
 *  }
 * Output:
 * for (i = 0; i < 1024; i ++) {
 *  A[i] = 0;
 * }
 * case3: Remoce the Let stmt. Substitute the var with value
 * Input:
 *  for (i = 0; i < 1024; i ++) {
 *    Let a = b + c + i;
 *    A[a - (b + c)] = 0;
 *  }
 * Output:
 * for (i = 0; i < 1024; i ++) {
 *  a[i] = 0;
 * }
 * case4: Set the init of For stmt to 0
 * Input:
 *  for (i = a + b; i < a + b + 1024; i ++) {
 *    A[i - (a + b)] = 0;
 *  }
 * Output:
 * for (i = 0; i < 1024; i ++) {
 *  A[i] = 0;
 * }
 */
StmtPtr simplifyStatement(StmtPtr stmt);
}  // namespace api

#endif  // ELENA_INCLUDE_PASS_COMMON_STATEMENTSIMPLIFY_H_
