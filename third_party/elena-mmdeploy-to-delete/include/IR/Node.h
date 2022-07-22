#ifndef ELENA_INCLUDE_IR_NODE_H_
#define ELENA_INCLUDE_IR_NODE_H_

#include <iostream>
#include <memory>

#include "Type.h"

namespace ir {
/**
 * @brief This is the base class of syntax analyze tree, and each of the
 * terminal and non-terminal is a node.
 *
 */
class Node {
 public:
  explicit Node(IRNodeType type);

  /**
   * @brief get Node type(such as Expr,Stmt,Array)
   * @author hanruobing
   */
  IRNodeType get_type() const;

  /**
   * @brief Get printable node type name.
   * @author xupengcheng
   * @return printable name for node type.
   */
  const char* get_type_name() const;

  virtual ~Node();

 private:
  IRNodeType type;
};
using NodePtr = std::shared_ptr<Node>;

template <typename N>
class NestedTypeNode : public Node {
 public:
  NestedTypeNode(N nested_type, IRNodeType type)
      : Node(type), nested_type(nested_type) {}

  N get_nested_type() const { return nested_type; }

 private:
  N nested_type;
};

#define IR_NODE_TYPE(Type) class Type;
#define IR_NODE_TYPE_NESTED(Type) \
  template <typename T>           \
  class Type;
#include "x/ir_node_types.def"

/**
 * cast_check is for dynamic type checking in ir::ptr_cast.
 */
namespace cast_check {

template <typename T>
inline bool is_compatible(IRNodeType x) {
  return x == T::type;
}

template <>
inline bool is_compatible<Node>(IRNodeType) {
  return true;
}

template <>
bool is_compatible<Var>(IRNodeType);

#define IR_NODE_TYPE_ABSTRACT(Type)                                      \
  template <>                                                            \
  inline bool is_compatible<Type>(IRNodeType x) {                        \
    constexpr bool is_casting_to_Expr = std::is_same<Expr, Type>::value; \
    switch (x) {
#define IR_NODE_TYPE_PLAIN(Type) case IRNodeType::Type:
#define IR_NODE_TYPE_NESTED(Type)
#define END_ABSTRACT                \
  return true;                      \
  case IRNodeType::Const:           \
  case IRNodeType::Array:           \
    return is_casting_to_Expr;      \
  default:                          \
    if (is_casting_to_Expr)         \
      return is_compatible<Var>(x); \
    else                            \
      return false;                 \
    }                               \
    }
#include "x/ir_node_types.def"

#define IR_NODE_TYPE_PLAIN(Type)                  \
  template <>                                     \
  inline bool is_compatible<Type>(IRNodeType x) { \
    return x == IRNodeType::Type;                 \
  }
#define IR_NODE_TYPE_ABSTRACT(Type)
#define IR_NODE_TYPE_NESTED(Type)
#include "x/ir_node_types.def"

template <typename T>
struct name_by_type;

template <>
struct name_by_type<Node> {
  static constexpr const char* value = "Node";
};

#define IR_NODE_TYPE(Type)                      \
  template <>                                   \
  struct name_by_type<Type> {                   \
    static constexpr const char* value = #Type; \
  };
#define IR_NODE_TYPE_NESTED(Type)               \
  template <typename A>                         \
  struct name_by_type<Type<A>> {                \
    static constexpr const char* value = #Type; \
  };
#include "x/ir_node_types.def"

}  // namespace cast_check

template <typename TP, typename UP>
std::shared_ptr<TP> ptr_cast(const std::shared_ptr<UP>& node) {
  if (!node) {
    ELENA_WARN("the code relies on 'ptr_cast(nullptr) == nullptr'.");
    return nullptr;
  }
  if (cast_check::is_compatible<TP>(node->get_type())) {
    return std::static_pointer_cast<TP>(node);
  } else {
    return nullptr;
  }
}

}  // namespace ir

#endif  // ELENA_INCLUDE_IR_NODE_H_
