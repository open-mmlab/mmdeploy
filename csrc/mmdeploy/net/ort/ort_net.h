// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_ORT_ORT_NET_H_
#define MMDEPLOY_SRC_NET_ORT_ORT_NET_H_

#include "mmdeploy/core/net.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

namespace mmdeploy::framework {

class OrtNet : public Net {
 public:
  ~OrtNet() override = default;
  Result<void> Init(const Value& cfg) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override;

 private:
  Ort::Env env_;
  Ort::Session session_{nullptr};
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  Device device_;
  Stream stream_;
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_ORT_ORT_NET_H_
