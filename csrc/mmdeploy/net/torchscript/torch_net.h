// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_NET_TORCHSCRIPT_TORCH_NET_H_
#define MMDEPLOY_CSRC_MMDEPLOY_NET_TORCHSCRIPT_TORCH_NET_H_

#include "mmdeploy/core/net.h"
#include "torch/script.h"

namespace mmdeploy {

class TorchNet : public Net {
 public:
  ~TorchNet() override;
  Result<void> Init(const Value& cfg) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override;

 private:
  Result<Tensor> FromTorchTensor(const torch::Tensor& tensor, const std::string& name);

  torch::jit::script::Module module_;
  std::vector<Tensor> input_tensor_;
  std::vector<Tensor> output_tensor_;
  Device device_;
  Stream stream_;
  std::optional<torch::Device> torch_device_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_NET_TORCHSCRIPT_TORCH_NET_H_
