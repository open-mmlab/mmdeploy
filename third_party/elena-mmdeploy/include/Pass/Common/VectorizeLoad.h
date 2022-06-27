#ifndef ELENA_INCLUDE_PASS_COMMON_VECTORIZELOAD_H_
#define ELENA_INCLUDE_PASS_COMMON_VECTORIZELOAD_H_

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

class Vectorizer : public MutatorBase<Vectorizer> {
 public:
  Vectorizer(ir::IterVarPtr var, int var_lanes)
      : var_(var), var_lanes_(var_lanes) {}

  using MutatorBase::visit;
  ir::NodePtr visit(ir::IfThenElse *node);
  ir::NodePtr visit(ir::Store *node);
  ir::NodePtr visit(ir::ScalarVar *node);
  ir::NodePtr visit(ir::IterVar *node);
  ir::NodePtr visit(ir::Select *node);
  ir::NodePtr visit(ir::Binary *node);

  bool if_vectorize{true};

 private:
  ir::IterVarPtr var_;
  int var_lanes_;
  ExprPtr ramp_;
  StmtCopy stmt_copyer;
};

class VectorizeLoader : public MutatorBase<VectorizeLoader> {
 public:
  using MutatorBase::visit;
  ir::NodePtr visit(ir::For *node);
};

namespace api {
ir::StmtPtr vectorizeLoad(ir::StmtPtr stmt);

}  // namespace api

#endif  // ELENA_INCLUDE_PASS_COMMON_VECTORIZELOAD_H_
