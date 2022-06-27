#ifndef ELENA_INCLUDE_IR_VISITORBASE_H_
#define ELENA_INCLUDE_IR_VISITORBASE_H_

#include "IR/Pass.h"

/** @brief General case for VisitorBase<Res>.
 * @author xieruifeng
 */
template <typename Derived, typename Res = void>
class VisitorBase : public Pass<Derived, Res> {};

/** @brief Special case for VisitorBase<Res = void>.
 * @author xieruifeng
 */
template <typename Derived>
class VisitorBase<Derived, void> : public Pass<Derived, void> {
 public:
/**
 * @brief See the general case for reference. In this version of VisitorBase the
 * visit functions have default implementation.
 */
#define IR_NODE_TYPE_PLAIN(Type) void visit(ir::Type *);
#define IR_NODE_TYPE_ABSTRACT(Type)
#define IR_NODE_TYPE_NESTED(Type) \
  template <typename T>           \
  void visit(ir::Type<T> *);
#include "x/ir_node_types.def"
  using Pass<Derived, void>::visit;

 protected:
  using Pass<Derived, void>::derived;
};

#define VISITOR VisitorBase<Derived, void>
#define VISIT(x) visit((x).get())
#define VISIT_RETURN(x) static_cast<void>(0)
#define RETURN_TYPE void
#include "x/PassImpl.def"

#endif  // ELENA_INCLUDE_IR_VISITORBASE_H_
