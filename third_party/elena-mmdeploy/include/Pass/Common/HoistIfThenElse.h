#ifndef ELENA_INCLUDE_PASS_COMMON_HOISTIFTHENELSE_H_
#define ELENA_INCLUDE_PASS_COMMON_HOISTIFTHENELSE_H_

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

using HoistMap = std::unordered_map<ir::Node*, std::vector<ir::StmtPtr>>;
using HoistVarMap = std::unordered_map<std::string, std::vector<ir::StmtPtr>>;
using VarMap = std::unordered_map<ir::Node*, std::unordered_set<ir::Node*>>;

class IfThenElseHoist {
 public:
  StmtPtr visitAndMutate(StmtPtr stmt) {
    selectCandidates(stmt);
    locateTopFor();
    return postOrderMutateStmt(stmt);
  }

 private:
  void selectCandidates(StmtPtr stmt);
  void locateTopFor();
  StmtPtr postOrderMutateStmt(StmtPtr stmt);
  size_t getUpdatedFor(StmtPtr for_stmt, StmtPtr if_stmt);
  StmtPtr hoistIf(StmtPtr if_stmt);

  // Map of all For nodes to all child IfThenElse nodes.
  HoistMap For2IfMap;
  // Map of all IfThenElse nodes to all For nodes which are loop invariant.
  HoistMap If2ForMap;
  // Map of highest loop invariant For to child IfThenElse.
  HoistMap TopForVarMap;
  // Map of original For to list of update For nodes.
  HoistMap ForTrackingMap;
  // Map of all IfThenElse nodes to condition variable nodes.
  VarMap CondVarMap;
  // List of For nodes added in post order DFS visiting.
  std::vector<ir::Stmt*> OrderedForList;
};

namespace api {
ir::StmtPtr hoistIfThenElse(ir::StmtPtr stmt);

}  // namespace api

#endif  // ELENA_INCLUDE_PASS_COMMON_HOISTIFTHENELSE_H_
