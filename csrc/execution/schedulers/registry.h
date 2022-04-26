//
// Created by zhangli on 4/26/22.
//

#ifndef MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_
#define MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_

#include "core/registry.h"
#include "execution/type_erased.h"

namespace mmdeploy {

namespace detail {

template <>
struct get_return_type<TypeErasedScheduler<Value>> {
  using type = TypeErasedScheduler<Value>;
};

}  // namespace detail

template <>
struct is_cast_by_erasure<TypeErasedScheduler<Value>> : std::true_type {};

MMDEPLOY_REGISTER_TYPE_ID(TypeErasedScheduler<Value>, 8);

MMDEPLOY_DECLARE_REGISTRY(TypeErasedScheduler<Value>);

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_
