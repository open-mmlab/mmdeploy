// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_PPL_PPL_NET_H_
#define MMDEPLOY_SRC_NET_PPL_PPL_NET_H_

#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/net.h"
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/runtime/runtime.h"

namespace mmdeploy::framework {

using PPLTensor = ppl::nn::Tensor;

class PPLNet : public Net {
 public:
  ~PPLNet() override;

  Result<void> Init(const Value& args) override;

  Result<void> Deinit() override;

  Result<void> Reshape(Span<TensorShape> input_shapes) override;

  Result<Span<Tensor> > GetInputTensors() override;

  Result<Span<Tensor> > GetOutputTensors() override;

  Result<void> Forward() override;

  Result<void> ForwardAsync(Event* event) override;

  static Result<std::vector<TensorShape> > InferOutputShapes(Span<TensorShape> input_shapes,
                                                             Span<TensorShape> prev_in_shapes,
                                                             Span<TensorShape> prev_out_shapes);

 private:
  static Tensor CreateInternalTensor(ppl::nn::Tensor* src, Device device);

  static Result<int64_t> GetBatchSize(Span<TensorShape> shapes);

  static std::vector<TensorShape> GetShapes(Span<Tensor> tensors);

  Device device_;
  Stream stream_;
  std::vector<std::unique_ptr<ppl::nn::Engine> > engines_;
  std::vector<Tensor> inputs_external_;
  std::vector<Tensor> outputs_external_;
  std::vector<PPLTensor*> inputs_internal_;
  std::vector<PPLTensor*> outputs_internal_;
  std::unique_ptr<ppl::nn::Runtime> runtime_;
  bool can_infer_output_shapes_{false};
  static constexpr const auto kHost = Device(0);
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_PPL_PPL_NET_H_
