// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_RKNN_RKNN_NET_H_
#define MMDEPLOY_SRC_NET_RKNN_RKNN_NET_H_

#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/net.h"
#include "rknn_api.h"

namespace mmdeploy::framework {

class RKNNNet : public Net {
 public:
  ~RKNNNet() override;

  Result<void> Init(const Value& args) override;

  Result<void> Deinit() override;

  Result<void> Reshape(Span<TensorShape> input_shapes) override;

  Result<Span<Tensor> > GetInputTensors() override;

  Result<Span<Tensor> > GetOutputTensors() override;

  Result<void> Forward() override;

  Result<void> ForwardAsync(Event* event) override;

 private:
  void PrintRKNNTensorAttr(const char* tag, const std::vector<rknn_tensor_attr>& attrs);

  Device device_;
  Stream stream_;
  rknn_context ctx_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  static constexpr const auto kHost = Device(0);
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_RKNN_RKNN_NET_H_
