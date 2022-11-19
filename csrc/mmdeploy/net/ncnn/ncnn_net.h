// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_NCNN_NCNN_NET_H_
#define MMDEPLOY_SRC_NET_NCNN_NCNN_NET_H_

#include "mmdeploy/core/net.h"
// It's ncnn's net.h
#include "net.h"

namespace mmdeploy::framework {

class NCNNNet : public Net {
 public:
  ~NCNNNet() override;
  Result<void> Init(const Value& args) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override { return success(); };

 private:
  Device device_;
  Stream stream_;
  std::string params_;
  std::string weights_;
  std::vector<int> input_indices_;
  std::vector<int> output_indices_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  ncnn::Net net_;
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_NCNN_NCNN_NET_H_
