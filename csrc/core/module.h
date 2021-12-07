// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MODULE_H_
#define MMDEPLOY_SRC_CORE_MODULE_H_

#include "core/macro.h"
#include "core/status_code.h"
#include "core/value.h"

namespace mmdeploy {

class MM_SDK_API Module {
 public:
  virtual ~Module() = default;
  virtual Result<Value> Process(const Value& args) = 0;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_MODULE_H_
