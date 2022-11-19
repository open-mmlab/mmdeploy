// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_NET_H_
#define MMDEPLOY_SRC_CORE_NET_H_

#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy::framework {

class Net {
 public:
  virtual ~Net() = default;
  virtual Result<void> Init(const Value& cfg) = 0;
  virtual Result<void> Deinit() = 0;
  virtual Result<Span<Tensor>> GetInputTensors() = 0;
  virtual Result<Span<Tensor>> GetOutputTensors() = 0;
  virtual Result<void> Reshape(Span<TensorShape> input_shapes) = 0;
  virtual Result<void> Forward() = 0;
  virtual Result<void> ForwardAsync(Event* event) = 0;
};

MMDEPLOY_DECLARE_REGISTRY(Net, std::unique_ptr<Net>(const Value& config));

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_CORE_NET_H_
