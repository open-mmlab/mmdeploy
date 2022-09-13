// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_
#define MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_

#include "mmdeploy/core/registry.h"
#include "mmdeploy/execution/type_erased.h"

namespace mmdeploy {

namespace detail {

template <>
struct get_return_type<TypeErasedScheduler<Value>> {
  using type = TypeErasedScheduler<Value>;
};

}  // namespace detail

MMDEPLOY_REGISTER_TYPE_ID(TypeErasedScheduler<Value>, 8);

MMDEPLOY_DECLARE_REGISTRY(TypeErasedScheduler<Value>);

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_
