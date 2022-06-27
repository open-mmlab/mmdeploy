#ifndef ELENA_INCLUDE_PASS_COMMON_INJECTVIRTUALTHREAD_H_
#define ELENA_INCLUDE_PASS_COMMON_INJECTVIRTUALTHREAD_H_

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

/**
 * Substitue itervar with Expr, use for Substitue vthread itervar
 * Now add For statement
 */
class IRSubstitue : public MutatorBase<IRSubstitue> {
 public:
  explicit IRSubstitue(const std::unordered_map<std::string, ir::ExprPtr>& smap)
      : smap_(smap) {}
  using MutatorBase::visit;

  ir::NodePtr visit(ir::IterVar* node) {
    auto var_ptr = node->shared_from_this();
    auto it = smap_.find(var_ptr->get_name());
    if (it != smap_.end()) {
      return it->second;
    } else {
      return var_ptr;
    }
  }

 private:
  const std::unordered_map<std::string, ir::ExprPtr>& smap_;
};

class VirtualThreadInjector : public MutatorBase<VirtualThreadInjector> {
 public:
  VirtualThreadInjector();
  ir::NodePtr mutateReplace(ir::NodePtr node);

  using MutatorBase::visit;
  ir::NodePtr visit(ir::Attr* node);
};

namespace api {
ir::StmtPtr injectVirtualThread(ir::StmtPtr stmt);

}  // namespace api

#endif  // ELENA_INCLUDE_PASS_COMMON_INJECTVIRTUALTHREAD_H_
