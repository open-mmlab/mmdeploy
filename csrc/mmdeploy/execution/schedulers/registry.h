// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_
#define MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/execution/type_erased.h"

namespace mmdeploy {

MMDEPLOY_REGISTER_TYPE_ID(TypeErasedScheduler<Value>, 8);

MMDEPLOY_DECLARE_REGISTRY(TypeErasedScheduler<Value>,
                          TypeErasedScheduler<Value>(const Value& config));

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_REGISTRY_H_
