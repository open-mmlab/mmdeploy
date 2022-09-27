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

namespace _maybe {

struct Maybe {
  std::optional<std::reference_wrapper<const Value>> val_;
  explicit operator bool() const noexcept { return val_.has_value(); }
  const Value& operator*() const noexcept { return val_->get(); }
  const Value* operator->() const noexcept { return &val_->get(); }
};

inline Maybe operator/(const Maybe& maybe, const string& p) {
  if (maybe && maybe->contains(p)) {
    return {(*maybe)[p]};
  }
  return {std::nullopt};
}

template <typename T>
inline std::optional<T> operator/(const Maybe& maybe, identity<T>) {
  if (maybe) {
    return maybe->get<T>();
  }
  return std::nullopt;
}
}  // namespace _maybe

using _maybe::Maybe;

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_GRAPH_COMMON_H_
