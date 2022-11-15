// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MODULE_H_
#define MMDEPLOY_SRC_CORE_MODULE_H_

#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

class MMDEPLOY_API Module {
 public:
  virtual ~Module() = default;
  virtual Result<Value> Process(const Value& args) = 0;
};

MMDEPLOY_DECLARE_REGISTRY(Module);

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_MODULE_H_
