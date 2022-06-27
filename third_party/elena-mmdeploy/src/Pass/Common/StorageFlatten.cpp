#include "Pass/Common/StorageFlatten.h"

#include <vector>

#include "IR/Expr.h"
#include "IR/Type.h"
#include "api.h"

using ir::BinaryType;
using ir::IRNodeType;
using ir::Node;
using ir::ScalarType;
StorageFlattener::StorageFlattener() {}

StorageFlattener::StorageFlattener(
    ir::MapPtr<ir::IterVar, ir::Range> bound_map_)
    : bound_map(bound_map_) {}

ir::NodePtr StorageFlattener::mutateReplace(ir::NodePtr node) {
  if (node == nullptr) {
    return node;
  }
  return visit(node.get());
}

ir::NodePtr StorageFlattener::visit(ir::Attr* attr) {
  auto key = attr->key;
  if (key == ir::AttrType::RealizeScope) {
    auto tensor_node_ptr = attr->node;
    auto tensor_ptr = ir::ptr_cast<ir::TensorVar, ir::Node>(tensor_node_ptr);
    auto body = attr->body;
    mutate(body);
    return std::make_shared<ir::Attr>(tensor_ptr, ir::AttrType::StorageScope,
                                      attr->value, body);
  } else {
    mutate(attr->body);
    return attr->shared_from_this();
  }
}

ir::NodePtr StorageFlattener::visit(ir::ScalarVar* scalar_ptr) {
  if (scalar_ptr->indices == nullptr) return scalar_ptr->shared_from_this();
  mutate(scalar_ptr->indices);

  auto tensor_ptr = scalar_ptr->tensor;
  if (origin_bias.count(tensor_ptr)) {
    // avoid flatten the indices which already flattened
    if (flattened_indices.count(scalar_ptr->indices)) {
      return std::make_shared<ir::ScalarVar>(
          scalar_ptr->tensor, scalar_ptr->indices,
          static_cast<ir::Var*>(scalar_ptr)->get_name());
    }
    std::vector<std::shared_ptr<ir::Expr>> vie;

    ExprPtr stride;
    ExprPtr accumulate;
    for (int idx = scalar_ptr->indices->element.size() - 1; idx > -1; idx--) {
      if (idx == scalar_ptr->indices->element.size() - 1) {
        auto x = scalar_ptr->indices->element[idx];
        auto y = origin_bias.at(tensor_ptr)[idx];
        accumulate = x - y;
        /*accumulate = (scalar_ptr->indices->element[idx] -
                      origin_bias.at(tensor_ptr)[idx]);*/
        stride = origin_extent.at(tensor_ptr)[idx];
      } else {
        accumulate = accumulate + stride * (scalar_ptr->indices->element[idx] -
                                            origin_bias.at(tensor_ptr)[idx]);
        stride = stride * origin_extent.at(tensor_ptr)[idx];
      }
    }
    vie.push_back(accumulate);
    auto new_index = std::make_shared<ir::Array<ir::Expr>>(vie);
    flattened_indices.insert(new_index);
    return std::make_shared<ir::ScalarVar>(
        scalar_ptr->tensor, new_index,
        static_cast<ir::Var*>(scalar_ptr)->get_name());
  } else {
    // this should be placeholder
    std::vector<std::shared_ptr<ir::Expr>> vie;
    ExprPtr stride;
    ExprPtr accumulate;
    for (int idx = scalar_ptr->indices->element.size() - 1; idx > -1; idx--) {
      if (idx == scalar_ptr->indices->element.size() - 1) {
        accumulate = (scalar_ptr->indices->element[idx]);
        stride = tensor_ptr->shape->element[idx];
      } else {
        accumulate = accumulate + stride * scalar_ptr->indices->element[idx];
        stride = stride * tensor_ptr->shape->element[idx];
      }
    }
    vie.push_back(accumulate);
    auto new_index = std::make_shared<ir::Array<ir::Expr>>(vie);
    return std::make_shared<ir::ScalarVar>(
        scalar_ptr->tensor, new_index,
        static_cast<ir::Var*>(scalar_ptr)->get_name());
  }
  return scalar_ptr->shared_from_this();
}

ir::NodePtr StorageFlattener::visit(ir::Realize* realize_ptr) {
  // TODO(xieruifeng) in Realize: make var a TensorVar, or add ELENA_ASSERT
  // here.
  auto tensor_ptr = ir::ptr_cast<ir::TensorVar>(realize_ptr->var);

  auto array_ptr = realize_ptr->bound;
  std::vector<ir::ExprPtr> bias;
  std::vector<ir::ExprPtr> extent;
  std::vector<ir::RangePtr> allocate_bound;
  auto accumulate = array_ptr->element[0]->extent;
  for (int i = 0; i < array_ptr->element.size(); i++) {
    auto range = array_ptr->element[i];
    bias.push_back(range->init);
    extent.push_back(range->extent);
    if (i != 0) {
      accumulate = accumulate * (range->extent);
    }
  }
  auto zero = std::make_shared<ir::Const<uint64_t>>(0, ir::ScalarType::UInt64);
  origin_bias[tensor_ptr] = bias;
  origin_extent[tensor_ptr] = extent;
  mutate(realize_ptr->body);
  allocate_bound.push_back(std::make_shared<ir::Range>(zero, accumulate));
  auto realize_stmt = std::make_shared<ir::Allocate>(
      tensor_ptr, std::make_shared<ir::Array<ir::Range>>(allocate_bound),
      realize_ptr->body);
  // Author: XuPing
  realize_stmt->is_output = realize_ptr->is_output;
  return realize_stmt;
}

ir::NodePtr StorageFlattener::visit(ir::Provide* old_provide_ptr) {
  mutate(old_provide_ptr->value);
  if (old_provide_ptr->index) {
    mutate(old_provide_ptr->index);
  }

  // TODO(ruobing): maybe we should use tensor var for provide, not var
  // or else please add an ELENA_ASSERT here.
  auto var_ptr = ir::ptr_cast<ir::TensorVar>(old_provide_ptr->var);
  if (origin_bias.count(var_ptr)) {
    std::vector<std::shared_ptr<ir::Expr>> vie;
    ExprPtr accumulate;
    ExprPtr stride;

    for (int i = old_provide_ptr->index->element.size() - 1; i > -1; i--) {
      auto old_bias = old_provide_ptr->index->element[i];
      if (i == old_provide_ptr->index->element.size() - 1) {
        accumulate = ((old_bias - origin_bias[var_ptr][i]));
        stride = origin_extent[var_ptr][i];
      } else {
        accumulate =
            accumulate + stride * ((old_bias - origin_bias[var_ptr][i]));
        stride = stride * origin_extent[var_ptr][i];
      }
    }

    vie.push_back(accumulate);
    auto new_index = std::make_shared<ir::Array<ir::Expr>>(vie);

    return std::make_shared<ir::Store>(var_ptr, old_provide_ptr->value,
                                       new_index);
  } else {
    return old_provide_ptr->shared_from_this();
  }
}
