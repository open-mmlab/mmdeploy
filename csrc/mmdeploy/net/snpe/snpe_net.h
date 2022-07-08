// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_NCNN_NCNN_NET_H_
#define MMDEPLOY_SRC_NET_NCNN_NCNN_NET_H_

#include "mmdeploy/core/net.h"
#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"

namespace mmdeploy {

class SNPENet : public Net {
 public:
  ~SNPENet() override;
  Result<void> Init(const Value& args) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override { return Status(eNotSupported); };

 private:
  Device device_;
  Stream stream_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;

  std::unique_ptr<zdl::SNPE::SNPE> snpe_;
  std::unique_ptr<zdl::DlContainer::IDlContainer> container_;

  std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputs_internal_;
  zdl::DlSystem::TensorMap input_tensor_map_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_NET_NCNN_NCNN_NET_H_
