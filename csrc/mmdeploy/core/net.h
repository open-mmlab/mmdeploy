// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_NET_H_
#define MMDEPLOY_SRC_CORE_NET_H_

#include "mpl/span.h"
#include "registry.h"
#include "tensor.h"
#include "value.h"

namespace mmdeploy {

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

MMDEPLOY_DECLARE_REGISTRY(Net);

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_NET_H_
