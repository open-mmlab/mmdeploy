#ifndef ELENA_INCLUDE_PASS_COMMON_UNROLL_H_
#define ELENA_INCLUDE_PASS_COMMON_UNROLL_H_

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
#include "Pass/Common/StmtCopy.h"
#include "api.h"

class UnrollFuller : public MutatorBase<UnrollFuller> {
 public:
  using MutatorBase::visit;
  ir::NodePtr visit(ir::For* node);

 private:
  StmtCopy stmt_copyer;
  int offset = 1;
};

class AutoUnroller : public MutatorBase<AutoUnroller> {
 public:
  explicit AutoUnroller(bool is_full_unroll) : is_full_unroll(is_full_unroll) {}
  using MutatorBase::visit;
  ir::NodePtr visit(ir::For* node);
  ir::NodePtr visit(ir::IfThenElse* node);
  ir::NodePtr visit(ir::Let* node);

  bool is_unroll = true;
  bool is_full_unroll = false;
};

namespace api {
ir::StmtPtr unrollFull(ir::StmtPtr stmt);
ir::StmtPtr autoUnroll(ir::StmtPtr stmt, bool is_full_unroll);

}  // namespace api

#endif  // ELENA_INCLUDE_PASS_COMMON_UNROLL_H_
