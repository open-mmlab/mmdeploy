#ifndef ELENA_INCLUDE_IR_CONTAINER_H_
#define ELENA_INCLUDE_IR_CONTAINER_H_

#include <initializer_list>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Node.h"
#include "Type.h"

namespace ir {
/**
 * @brief Array for holding IR nodes.
 *
 * Note that we only support IR nodes (Node) to be contained in Array here, that
 * is, classes that have the ::type field of IRNodeType type.  This is required
 * to properly dispatch an Array node during visit.
 *
 * A static_assert has been performed and will provide a human-readable error
 * message when a type that's not supported was passed in.
 *
 * @author hanruobing, xupengcheng
 */
template <typename TNode>
class Array : public NestedTypeNode<IRNodeType>,
              public std::enable_shared_from_this<Array<TNode>> {
  template <typename T, typename = const IRNodeType>
  struct validity_check : std::false_type {};
  template <typename T>
  struct validity_check<T, decltype(T::type)> : std::true_type {};
  static_assert(validity_check<TNode>::value,
                "Array<TNode> only supports Node and subclasses");

 public:
  static const IRNodeType type = IRNodeType::Array;
  Array() : NestedTypeNode(TNode::type, type) {}

  Array(std::initializer_list<TNode> l) : NestedTypeNode(TNode::type, type) {
    for (const auto &e : l) {
      element.push_back(std::make_shared<TNode>(e));
    }
  }
  explicit Array(const std::vector<TNode> &l)
      : NestedTypeNode(TNode::type, type) {
    for (const auto &e : l) {
      element.push_back(std::make_shared<TNode>(e));
    }
  }
  Array(std::initializer_list<std::shared_ptr<TNode>> l)
      : NestedTypeNode(TNode::type, type) {
    for (const auto &e : l) {
      element.push_back(e);
    }
  }
  explicit Array(std::vector<std::shared_ptr<TNode>> l)
      : NestedTypeNode(TNode::type, type) {
    for (const auto &e : l) {
      element.push_back(e);
    }
  }
  std::shared_ptr<TNode> &operator[](size_t index) { return element[index]; }

  size_t findvar(std::shared_ptr<TNode> e) {
    size_t size = element.size();
    for (size_t i = 0; i < size; i++) {
      if (element[i] == e) return i;
    }
    return size;
  }

  size_t size() { return element.size(); }
  std::vector<std::shared_ptr<TNode>> element;
};
template <typename TNode>
using ArrayPtr = std::shared_ptr<Array<TNode>>;

template <typename KNode, typename TNode>
class Map {
 public:
  Map() {}
  std::unordered_map<std::shared_ptr<KNode>, std::shared_ptr<TNode>> element;
};
template <typename KNode, typename TNode>
using MapPtr = std::shared_ptr<Map<KNode, TNode>>;

}  // namespace ir
#endif  // ELENA_INCLUDE_IR_CONTAINER_H_
