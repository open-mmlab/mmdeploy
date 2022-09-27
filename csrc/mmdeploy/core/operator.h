// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_OPERATOR_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_OPERATOR_H_

#include "mmdeploy/core/value.h"

namespace mmdeploy::graph {

using std::string;
using std::tuple;
using std::vector;

MMDEPLOY_API Result<void> Gather(const Value::Array& array, const vector<int>& idxs,
                                 Value::Array& output);
MMDEPLOY_API Result<void> Gather(Value::Array&& array, const vector<int>& idxs,
                                 Value::Array& output);
MMDEPLOY_API Result<void> Gather(const Value::Object& object, const vector<std::string>& keys,
                                 Value::Array& output);
MMDEPLOY_API Result<void> Gather(Value::Object&& object, const vector<std::string>& keys,
                                 Value::Array& output);
MMDEPLOY_API Result<void> Scatter(Value::Array array, const vector<int>& idxs,
                                  Value::Array& output);
MMDEPLOY_API Result<void> Scatter(Value::Array array, const vector<std::string>& keys,
                                  Value::Object& output);

inline Result<Value::Array> Gather(const Value::Array& array, const vector<int>& idxs) {
  Value::Array output;
  OUTCOME_TRY(Gather(array, idxs, output));
  return output;
}

inline Result<Value::Array> Gather(Value::Array&& array, const vector<int>& idxs) {
  Value::Array output;
  OUTCOME_TRY(Gather(std::move(array), idxs, output));
  return output;
}

inline Result<Value::Array> Gather(const Value::Object& object, const vector<std::string>& keys) {
  Value::Array output;
  OUTCOME_TRY(Gather(object, keys, output));
  return output;
}

inline Result<Value::Array> Gather(Value::Object&& object, const vector<std::string>& keys) {
  Value::Array output;
  OUTCOME_TRY(Gather(std::move(object), keys, output));
  return output;
}

inline Result<Value::Array> Scatter(Value::Array array, const vector<int>& idxs) {
  Value::Array output(idxs.size(), Value::kNull);
  OUTCOME_TRY(Scatter(std::move(array), idxs, output));
  return output;
}

inline Result<Value::Object> Scatter(Value::Array array, const vector<std::string>& keys) {
  Value::Object output;
  OUTCOME_TRY(Scatter(std::move(array), keys, output));
  return output;
}

template <class V, std::enable_if_t<is_value_v<std::decay_t<V>>, bool> = true>
Result<tuple<Value, vector<int>>> Flatten(V&& input) {
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
  idxs.push_back(static_cast<int>(input.size()));
  return {output, idxs};
}

template <class V, std::enable_if_t<is_value_v<std::decay_t<V>>, bool> = true>
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
MMDEPLOY_API Result<Value> DistribOA(const Value& oa);

// array of objects -> object of arrays, all objects must be isomorphic
MMDEPLOY_API Result<Value> DistribAO(const Value& ao);

// array of arrays -> array of arrays, this is equivalent to transpose
MMDEPLOY_API Result<Value> DistribAA(const Value& a);

MMDEPLOY_API std::tuple<Value::Array, std::vector<int>> FlattenArray(Value::Array values,
                                                                     const vector<bool>& predicate);

MMDEPLOY_API Value::Array UnflattenArray(Value::Array values, const vector<int>& index,
                                         const vector<bool>& predicate);

MMDEPLOY_API Value::Array BroadcastArray(Value::Array values, const std::vector<int>& index,
                                         const vector<bool>& predicate);

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_OPERATOR_H_
