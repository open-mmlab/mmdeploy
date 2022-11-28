// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_SNPE_SNPE_NET_H_
#define MMDEPLOY_SRC_NET_SNPE_SNPE_NET_H_

#include <iostream>
#include <memory>
#include <string>

#include "DiagLog/IDiagLog.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/PlatformConfig.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "mmdeploy/core/net.h"

namespace mmdeploy::framework {

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
  void Build(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
             zdl::DlSystem::Runtime_t runtime, zdl::DlSystem::RuntimeList runtimeList,
             bool useUserSuppliedBuffers, zdl::DlSystem::PlatformConfig platformConfig);

  std::string ShapeStr(zdl::DlSystem::ITensor* pTensor);

  void copy_output(const zdl::DlSystem::ITensor* from, Tensor& to);
  void copy_input(const Tensor& from, zdl::DlSystem::ITensor* to);

  Device device_;
  Stream stream_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;

  std::unique_ptr<zdl::SNPE::SNPE> snpe_;
  std::unique_ptr<zdl::DlContainer::IDlContainer> container_;

  std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputs_internal_;
  zdl::DlSystem::TensorMap input_tensor_map_;
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_SNPE_SNPE_NET_H_
