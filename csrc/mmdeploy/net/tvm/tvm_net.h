// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_TVM_TVM_NET_H_
#define MMDEPLOY_SRC_NET_TVM_TVM_NET_H_

#include <tvm/runtime/module.h>

#include "mmdeploy/core/net.h"

namespace mmdeploy::framework {

class TVMNet : public Net {
 public:
  ~TVMNet() override = default;
  Result<void> Init(const Value& cfg) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override;

 private:
  tvm::runtime::Module mod_factory_;

  tvm::runtime::PackedFunc func_set_input_;
  tvm::runtime::PackedFunc func_get_output_;
  tvm::runtime::PackedFunc func_run_;
  bool use_vm_;

  std::map<std::string, int> input_ids_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  Device device_;
  Stream stream_;
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_TVM_TVM_NET_H_
