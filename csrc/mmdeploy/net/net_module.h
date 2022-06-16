// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_MODULE_NET_MODULE_H_
#define MMDEPLOY_SRC_MODULE_NET_MODULE_H_

#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

class NetModule {
 public:
  ~NetModule();
  NetModule(NetModule&&) noexcept;

  explicit NetModule(const Value& args);
  Result<Value> operator()(const Value& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_MODULE_NET_MODULE_H_
