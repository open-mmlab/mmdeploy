#ifndef ELENA_INCLUDE_IR_PASS_H_
#define ELENA_INCLUDE_IR_PASS_H_

#include <iostream>
#include <memory>

#include "IR/Expr.h"
#include "IR/Node.h"
#include "Schedule/Schedule.h"
#include "Stage.h"
#include "Stmt.h"
#include "Type.h"

namespace detail {
template <typename Res>
struct DefaultRetval {
  template <typename T>
  struct AlwaysFalse : std::false_type {};

  static_assert(AlwaysFalse<Res>::value,
                "No proper default return value specified.");
};

template <>
struct DefaultRetval<void> {
  static void get() {}
};

template <typename T>
struct DefaultRetval<T *> {
  static T *get() { return nullptr; }
};

template <typename T>
struct DefaultRetval<std::shared_ptr<T>> {
  static std::shared_ptr<T> get() { return nullptr; }
};
}  // namespace detail

/** @brief Base class for all visiting operations over the IR tree that does not
 * modify the nodes.
 * @author xupengcheng, xieruifeng
 */
template <typename Derived, typename Res = void>
class Pass {
 public:
/**
 * @brief Visit function for all kinds of IR nodes.  Function for each type of
 * IR node shall reveal the internal structure of the visited node.
 * @author xupengcheng
 * @param node node to be visited.
 */
#define IR_NODE_TYPE_PLAIN(Type) Res visit(ir::Type *node) = delete;
// It is DELIBERATE that visiting abstract nodes is NOT deleted.
// This will allow e.g. 'visit(p.get())' for 'p' being some 'ExprPtr'.
#define IR_NODE_TYPE_ABSTRACT(Type)
#define IR_NODE_TYPE_NESTED(Type) \
  template <typename T>           \
  Res visit(ir::Type<T> *node) = delete;
#include "x/ir_node_types.def"

  template <typename T>
  Res visit(const std::shared_ptr<T> &pnode) {
    derived().visit(pnode.get());
  }

  /**
   * @brief Recursively visit provided node.  This function dispatches according
   * to the actual type (IRNodeType stored in the node) of the node provided.
   * Subclasses may override this function to inject operations (e.g. print
   * details of the node), but the overriden implementation shall call its base
   * to recurse correctly.  Go to IRPrinter implementation for demonstration.
   * @author xupengcheng
   * @param node the node to be visited recursively.
   */
  Res visit(ir::Node *node) {
    using namespace ir;  // NOLINT
    if (!node) return detail::DefaultRetval<Res>::get();
    switch (node->get_type()) {
// All plain node types here.
#define IR_NODE_TYPE_ABSTRACT(Type)
#define IR_NODE_TYPE_PLAIN(Type) \
  case IRNodeType::Type:         \
    return derived().visit(static_cast<Type *>(node));
#include "x/ir_node_types.def"
      // Array<T> types here: T is a plain node type.
      case IRNodeType::Array: {
        auto arr = static_cast<NestedTypeNode<IRNodeType> *>(node);
        switch (arr->get_nested_type()) {
#define IR_NODE_TYPE_NESTED(Type)
#define IR_NODE_TYPE(Type) \
  case IRNodeType::Type:   \
    return derived().visit(static_cast<Array<Type> *>(arr));
#include "x/ir_node_types.def"
          default:
            ELENA_ABORT("Type '" << IRNODETYPE_NAME(arr->get_nested_type())
                                 << "' is not permitted in Array.");
        }
      }
      // Const<T> types here: T is a scalar type.
      case IRNodeType::Const: {
        auto expr = static_cast<Expr *>(node);
        switch (expr->get_dtype()) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type:                                 \
    return derived().visit(static_cast<Const<native_type> *>(expr));
#include "x/scalar_types.def"
        }
      }
      default:
        ELENA_ABORT("Abstract node type '" << node->get_type_name()
                                           << "' shall never be visited.");
    }
  }

 protected:
  Derived &derived() { return *static_cast<Derived *>(this); }
};

#endif  // ELENA_INCLUDE_IR_PASS_H_
