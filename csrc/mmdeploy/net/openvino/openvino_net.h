// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_OPENVINO_OPENVINO_NET_H_
#define MMDEPLOY_SRC_NET_OPENVINO_OPENVINO_NET_H_

#include "inference_engine.hpp"
#include "mmdeploy/core/net.h"

namespace mmdeploy::framework {

class OpenVINONet : public Net {
 public:
  ~OpenVINONet() override = default;
  Result<void> Init(const Value& cfg) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override;

 private:
  InferenceEngine::Core core_;
  InferenceEngine::CNNNetwork network_;
  InferenceEngine::InferRequest request_;
  std::map<std::string, std::string> net_config_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  std::string device_str_;
  Device device_;
  Stream stream_;
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_OPENVINO_OPENVINO_NET_H_
