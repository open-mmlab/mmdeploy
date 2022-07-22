#ifndef ELENA_INCLUDE_IR_MUTATORBASE_H_
#define ELENA_INCLUDE_IR_MUTATORBASE_H_

#include <memory>

#include "Pass.h"

/** @brief General case for MutatorBase<Res>.
 * @author xieruifeng
 */
template <typename Derived>
class MutatorBase : public Pass<Derived, ir::NodePtr> {
 public:
/**
 * @brief See the general case for reference. In this version of MutatorBase the
 * visit functions have default implementation.
 */
#define IR_NODE_TYPE_PLAIN(Type) ir::NodePtr visit(ir::Type *);
#define IR_NODE_TYPE_ABSTRACT(Type)
#define IR_NODE_TYPE_NESTED(Type) \
  template <typename T>           \
  ir::NodePtr visit(ir::Type<T> *);
#include "x/ir_node_types.def"
  using Pass<Derived, ir::NodePtr>::visit;

  template <typename T>
  void mutate(std::shared_ptr<T> &n) {
    n = ir::ptr_cast<T>(derived().visit(n.get()));
  }

  template <typename T>
  void DefaultMutate(std::shared_ptr<T> &n) {
    n = ir::ptr_cast<T>(visit(n.get()));
  }

 protected:
  using Pass<Derived, ir::NodePtr>::derived;
};

#define VISITOR MutatorBase<Derived>
#define VISIT mutate
#define VISIT_RETURN(x) return x->shared_from_this();
#define RETURN_TYPE ir::NodePtr
#include "x/PassImpl.def"

#endif  // ELENA_INCLUDE_IR_MUTATORBASE_H_
