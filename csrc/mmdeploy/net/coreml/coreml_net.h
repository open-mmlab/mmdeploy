// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_COREML_COREML_NET_H_
#define MMDEPLOY_SRC_NET_COREML_COREML_NET_H_

#include "mmdeploy/core/net.h"

namespace mmdeploy::framework {

namespace coreml {
class Execution;
}  // namespace coreml

class CoreMLNet : public Net {
 public:
  ~CoreMLNet() override = default;
  Result<void> Init(const Value& cfg) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override;

 private:
  std::unique_ptr<coreml::Execution> execution_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  Device device_;
  Stream stream_;

  friend class coreml::Execution;
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_ORT_ORT_NET_H_
