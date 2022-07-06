#ifndef ELENA_INCLUDE_IR_VARREPLACER_H_
#define ELENA_INCLUDE_IR_VARREPLACER_H_

#include <memory>
#include <unordered_map>

#include "IR/MutatorBase.h"

class VarReplacer : public MutatorBase<VarReplacer> {
 public:
  explicit VarReplacer(
      const std::unordered_map<ir::NodePtr, ir::NodePtr>& vmap);

  ir::NodePtr mutateReplace(ir::NodePtr node);
  ir::NodePtr visit(ir::IterVar* node);
  ir::NodePtr visit(ir::ScalarVar* scalar_var_ptr);
  using MutatorBase::visit;
  // TODO(lixiuhong): explain what exactly is happening here?
  // this class is named VarReplacer, but it replaces whatever node matches ...
  // if whoever wants to fix this, please refer to TensorReplacer
  //  template <typename T>
  //  void mutate(std::shared_ptr<T>& node) {
  //    auto it = replace_map.find(node);
  //    if (it != replace_map.end())
  //      node = ir::ptr_cast<T>(it->second);
  //    else
  //      default_mutate(node);
  //  }

 private:
  std::unordered_map<ir::NodePtr, ir::NodePtr> replace_map;
};

#endif  // ELENA_INCLUDE_IR_VARREPLACER_H_
