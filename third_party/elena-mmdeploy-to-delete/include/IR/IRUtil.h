#ifndef ELENA_INCLUDE_IR_IRUTIL_H_
#define ELENA_INCLUDE_IR_IRUTIL_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "IR/Expr.h"
#include "IR/IntSet.h"
#include "Node.h"
#include "Schedule/Schedule.h"
#include "Stage.h"
#include "Stmt.h"
#include "Type.h"

using ir::ExprPtr;
using ir::StmtPtr;

StmtPtr mergeNest(std::vector<StmtPtr> nest, StmtPtr body);
StmtPtr mergeNest(StmtPtr nest, StmtPtr body);

IntSetPtr expr2IntSet(ExprPtr expr,
                      std::unordered_map<ir::IterVarPtr, IntSetPtr> up_state_);

void propBoundToInputs(
    ExprPtr expr, std::unordered_map<ir::IterVarPtr, IntSetPtr>* up_state,
    ir::TensorVarMap<std::vector<std::vector<IntSetPtr>>>* tmap);

#endif  // ELENA_INCLUDE_IR_IRUTIL_H_
