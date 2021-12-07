// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_OPERATOR_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_OPERATOR_H_

#include "core/value.h"

namespace mmdeploy::graph {

using std::string;
using std::tuple;
using std::vector;

template <class V, std::enable_if_t<is_value_v<std::decay_t<V> >, bool> = true>
Result<void> Idxs2Keys(V&& array, const vector<string>& keys, Value& object) {
  if (!std::forward<V>(array).is_array() || std::forward<V>(array).size() < keys.size()) {
    return Status(eInvalidArgument);
  }
  if (!(object.is_null() || object.is_object())) {
    return Status(eInvalidArgument);
  }
  for (int i = 0; i < keys.size(); ++i) {
    object[keys[i]] = std::forward<V>(array)[i];
  }
  return success();
}

template <class V, std::enable_if_t<is_value_v<std::decay_t<V> >, bool> = true>
Result<Value> Idxs2Keys(V&& array, const vector<string>& keys) {
  Value object = ValueType::kObject;
  OUTCOME_TRY(Idxs2Keys(std::forward<V>(array), keys, object));
  return object;
}

template <class V, std::enable_if_t<is_value_v<std::decay_t<V> >, bool> = true>
Result<void> Keys2Idxs(V&& object, const vector<string>& keys, Value& array) {
  if (!std::forward<V>(object).is_object()) {
    return Status(eInvalidArgument);
  }
  if (!(array.is_null() || array.is_array())) {
    return Status(eInvalidArgument);
  }
  try {
    for (const auto& key : keys) {
      array.push_back(std::forward<V>(object)[key]);
    }
  } catch (...) {
    // TODO: forward exception
    return Status(eInvalidArgument);
  }
  return success();
}

template <class V, std::enable_if_t<is_value_v<std::decay_t<V> >, bool> = true>
Result<Value> Keys2Idxs(V&& object, const vector<string>& keys) {
  Value array = ValueType::kArray;
  OUTCOME_TRY(Keys2Idxs(std::forward<V>(object), keys, array));
  return array;
}

template <class V, std::enable_if_t<is_value_v<std::decay_t<V> >, bool> = true>
Result<tuple<Value, vector<int> > > Flatten(V&& input) {
  if (!input.is_array()) {
    return Status(eInvalidArgument);
  }
  Value output = ValueType::kArray;
  std::vector<int> idxs;
  for (int i = 0; i < input.size(); ++i) {
    auto inner = std::forward<V>(input)[i];
    if (!inner.is_array()) {
      return Status(eInvalidArgument);
    }
    for (auto& item : inner) {
      output.push_back(std::move(item));
      idxs.push_back(i);
    }
  }
  idxs.push_back(input.size());
  return {output, idxs};
}

template <class V, std::enable_if_t<is_value_v<std::decay_t<V> >, bool> = true>
Result<Value> Unflatten(V&& input, const vector<int>& idxs) {
  if (!input.is_array()) {
    return Status(eInvalidArgument);
  }
  Value output = ValueType::kArray;
  for (int i = 0; i < idxs.back(); ++i) {
    output.push_back(ValueType::kArray);
  }
  for (int i = 0; i < input.size(); ++i) {
    if (idxs[i] >= output.size()) {
      return Status(eInvalidArgument);
    }
    output[idxs[i]].push_back(std::forward<V>(input)[i]);
  }
  return output;
}

// object of arrays -> array of objects, all arrays must be of same length
Result<Value> DistribOA(const Value& oa);

// array of objects -> object of arrays, all objects must be isomorphic
Result<Value> DistribAO(const Value& ao);

// array of arrays -> array of arrays, this is equivalent to transpose
Result<Value> DistribAA(const Value& a);

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_OPERATOR_H_
