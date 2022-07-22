#ifndef ELENA_INCLUDE_SCHEDULE_TENSORREPLACER_H_
#define ELENA_INCLUDE_SCHEDULE_TENSORREPLACER_H_

#include <string>
#include <unordered_map>

#include "IR/MutatorBase.h"

class TensorReplacer : public MutatorBase<TensorReplacer> {
 public:
  explicit TensorReplacer(const ir::TensorVarMap<ir::TensorVarPtr>& tmap);

  ir::NodePtr MutateReplace(ir::NodePtr node);
  ir::NodePtr visit(ir::TensorVar* node);
  using MutatorBase::visit;

 private:
  ir::TensorVarMap<ir::TensorVarPtr> replace_map;
};

#endif  // ELENA_INCLUDE_SCHEDULE_TENSORREPLACER_H_
