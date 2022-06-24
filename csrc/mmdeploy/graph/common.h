// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_GRAPH_COMMON_H_
#define MMDEPLOY_SRC_GRAPH_COMMON_H_

#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy::graph {

namespace {

template <typename T>
inline auto Check(const T& v) -> decltype(!!v) {
  return !!v;
}

template <typename T>
inline std::true_type Check(T&&) {
  return {};
}

}  // namespace

Result<std::vector<std::string>> ParseStringArray(const Value& value);

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_GRAPH_COMMON_H_
