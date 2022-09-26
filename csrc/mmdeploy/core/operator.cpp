// Copyright (c) OpenMMLab. All rights reserved.

#include "operator.h"

#include <algorithm>

#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy::graph {

Result<void> Gather(const Value::Array& array, const vector<int>& idxs, Value::Array& output) {
  if (idxs.empty()) {
    return success();
  }
  auto max_idx = *max_element(begin(idxs), end(idxs));
  if (array.size() <= max_idx) {
    return Status(eOutOfRange);
  }
  output.reserve(output.size() + idxs.size());
  for (const auto& idx : idxs) {
    output.push_back(array[idx]);
  }
  return success();
}

Result<void> Gather(Value::Array&& array, const vector<int>& idxs, Value::Array& output) {
  if (idxs.empty()) {
    return success();
  }
  auto max_idx = *max_element(begin(idxs), end(idxs));
  if (array.size() <= max_idx) {
    return Status(eOutOfRange);
  }
  output.reserve(output.size() + idxs.size());
  for (const auto& idx : idxs) {
    output.push_back(std::move(array[idx]));
  }
  return success();
}

Result<void> Gather(const Value::Object& object, const vector<std::string>& keys,
                    Value::Array& output) {
  output.reserve(output.size() + keys.size());
  try {
    for (const auto& key : keys) {
      output.push_back(object.at(key));
    }
  } catch (const std::out_of_range& e) {
    return Status(eOutOfRange);
  }
  return success();
}

Result<void> Gather(Value::Object&& object, const vector<std::string>& keys, Value::Array& output) {
  output.reserve(output.size() + keys.size());
  try {
    for (const auto& key : keys) {
      output.push_back(std::move(object.at(key)));
    }
  } catch (const std::out_of_range& e) {
    return Status(eOutOfRange);
  }
  return success();
}

Result<void> Scatter(Value::Array array, const vector<int>& idxs, Value::Array& output) {
  if (array.size() < idxs.size()) {
    return Status(eOutOfRange);
  }
  for (int i = 0; i < idxs.size(); ++i) {
    output[idxs[i]] = std::move(array[i]);
  }
  return success();
}

Result<void> Scatter(Value::Array array, const vector<std::string>& keys, Value::Object& output) {
  if (array.size() < keys.size()) {
    return Status(eOutOfRange);
  }
  for (int i = 0; i < keys.size(); ++i) {
    output.emplace(keys[i], std::move(array[i]));
  }
  return success();
}

Result<Value> DistribOA(const Value& oa) {
  if (!oa.is_object()) {
    return Status(eInvalidArgument);
  }
  Value ao = ValueType::kArray;
  for (auto inner = oa.begin(); inner != oa.end(); ++inner) {
    if (!inner->is_array()) {
      return Status(eInvalidArgument);
    }
    if (ao.empty()) {
      for (int i = 0; i < inner->size(); ++i) ao.push_back(ValueType::kObject);
    }
    if (inner->size() != oa.size()) {
      return Status(eInvalidArgument);
    }
    for (int i = 0; i < inner->size(); ++i) {
      ao[i][inner.key()] = (*inner)[i];
    }
  }
  return ao;
}

Result<Value> DistribAO(const Value& ao) {
  if (!ao.is_array()) {
    return Status(eInvalidArgument);
  }
  Value oa = ValueType::kObject;
  for (const auto& inner : ao) {
    if (inner.is_object()) {
      return Status(eInvalidArgument);
    }
    if (oa.empty()) {
      for (auto item = inner.begin(); item != inner.end(); ++item) {
        oa[item.key()] = ValueType::kObject;
      }
    }
    if (inner.size() != oa.size()) {
      return Status(eInvalidArgument);
    }
    for (auto item = inner.begin(); item != inner.end(); ++item) {
      if (!oa.contains(item.key())) {
        return Status(eInvalidArgument);
      }
      oa[item.key()].push_back(*item);
    }
  }
  return oa;
}

Result<Value> DistribAA(const Value& a) {
  if (!a.is_array()) {
    return Status(eInvalidArgument);
  }
  auto ta = Value::Array{};
  for (const auto& inner : a.get_ref<const Value::Array&>()) {
    if (!inner.is_array()) {
      return Status(eInvalidArgument);
    }
    if (ta.empty()) {
      ta.reserve(inner.size());
      for (int i = 0; i < inner.size(); ++i) {
        ta.emplace_back(Value::kArray);
      }
    }
    if (inner.size() != ta.size()) {
      return Status(eInvalidArgument);
    }
    for (int i = 0; i < inner.size(); ++i) {
      ta[i].push_back(inner[i]);
    }
  }
  return ta;
}

std::tuple<Value::Array, std::vector<int>> FlattenArray(Value::Array values,
                                                        const vector<bool>& predicate) {
  assert(values.size() == predicate.size());
  std::vector<int> indices;
  for (int i = 0; i < values.size(); ++i) {
    if (predicate[i]) {
      std::vector<int> idx;
      std::tie(values[i], idx) = Flatten(values[i]).value();
      if (indices.empty()) {
        indices.swap(idx);
      } else {
        assert(idx == indices);
      }
    }
  }
  return {std::move(values), std::move(indices)};
}

Value::Array UnflattenArray(Value::Array values, const vector<int>& index,
                            const vector<bool>& predicate) {
  assert(values.size() == predicate.size());
  for (int i = 0; i < values.size(); ++i) {
    if (predicate[i]) {
      values[i] = Unflatten(std::move(values[i]), index).value();
    }
  }
  return values;
}

Value::Array BroadcastArray(Value::Array values, const vector<int>& index,
                            const vector<bool>& predicate) {
  assert(values.size() == predicate.size());
  for (int i = 0; i < values.size(); ++i) {
    if (predicate[i]) {
      assert(values[i].is_array());
      auto& val = values[i].array();
      Value::Array ret(index.size() - 1);
      for (int j = 0; j < ret.size(); ++j) {
        ret[j] = val[index[j]];
      }
      values[i] = std::move(ret);
    }
  }
  return values;
}

}  // namespace mmdeploy::graph
