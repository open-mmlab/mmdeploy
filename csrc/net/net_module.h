// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_MODULE_NET_MODULE_H_
#define MMDEPLOY_SRC_MODULE_NET_MODULE_H_

#include <thread>
#include <vector>

#include "archive/value_archive.h"
#include "core/logger.h"
#include "core/model.h"
#include "core/module.h"
#include "core/net.h"
#include "core/registry.h"
#include "core/status_code.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "core/utils/scope_counter.h"
#include "core/value.h"
#include "experimental/module_adapter.h"

namespace mmdeploy {

class NetModule {
 public:
  ~NetModule();
  explicit NetModule(const Value& args);
  NetModule(NetModule&&) = default;
  Result<Value> operator()(const Value& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_MODULE_NET_MODULE_H_
