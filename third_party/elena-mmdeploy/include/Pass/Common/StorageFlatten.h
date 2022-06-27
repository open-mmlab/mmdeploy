#ifndef ELENA_INCLUDE_PASS_COMMON_STORAGEFLATTEN_H_
#define ELENA_INCLUDE_PASS_COMMON_STORAGEFLATTEN_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unordered_set>

#include "IR/Container.h"
#include "IR/Expr.h"
#include "IR/MutatorBase.h"
/**
 * @brief flattenStorage pass for IR
 * @author xiaoquanlun
 * @details split multi-dimension vars into single-dimension
 */

class StorageFlattener : public MutatorBase<StorageFlattener> {
 public:
  StorageFlattener();
  explicit StorageFlattener(ir::MapPtr<ir::IterVar, ir::Range> bound_map_);

  ir::NodePtr mutateReplace(ir::NodePtr node);

  ir::NodePtr visit(ir::Realize* node);
  ir::NodePtr visit(ir::Provide* node);
  ir::NodePtr visit(ir::ScalarVar* node);
  ir::NodePtr visit(ir::Attr* node);
  using MutatorBase::visit;

 private:
  ir::TensorVarMap<std::vector<ir::ExprPtr>> origin_bias;
  ir::TensorVarMap<std::vector<ir::ExprPtr>> origin_extent;

  ir::MapPtr<ir::IterVar, ir::Range> bound_map;
  std::unordered_set<ir::ArrayPtr<ir::Expr>> flattened_indices;
};

#endif  // ELENA_INCLUDE_PASS_COMMON_STORAGEFLATTEN_H_
