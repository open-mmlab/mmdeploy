// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_IPU_NET_H_
#define MMDEPLOY_SRC_NET_IPU_NET_H_

#include <boost/program_options.hpp>
#include <iostream>
#include <memory>
#include <model_runtime/ModelRunner.hpp>
#include <model_runtime/Tensor.hpp>
#include <popef/Types.hpp>
#include <string>

#include "mmdeploy/core/net.h"
#include "utils.hpp"

namespace mmdeploy {

class IPUNet : public Net {
 public:
  ~IPUNet() override;
  Result<void> Init(const Value& args) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override { return Status(eNotSupported); };

  void copy_output(const model_runtime::TensorMemory* from, Tensor& to, popef::DataType& dtype);
  void copy_input(const Tensor& from, model_runtime::TensorMemory* to, popef::DataType& dtype);

 private:
  model_runtime::ModelRunnerConfig config;
  model_runtime::ModelRunner model_runner;
  model_runtime::InputMemory input_memory;
  model_runtime::OutputMemory output_memory;
  std::vector<model_runtime::DataDesc> input_desc;
  std::vector<model_runtime::DataDesc> output_desc;

  Device device_;
  Stream stream_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_NET_IPU_NET_H_
