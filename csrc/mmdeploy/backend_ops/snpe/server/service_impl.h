// Copyright (c) OpenMMLab. All rights reserved.

#ifndef SERVICE_IMPL_H
#define SERVICE_IMPL_H

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

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
#include "inference.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using mmdeploy::Empty;
using mmdeploy::Inference;
using mmdeploy::Model;
using mmdeploy::Reply;
using mmdeploy::Tensor;
using mmdeploy::TensorList;

// Logic and data behind the server's behavior.
class InferenceServiceImpl final : public Inference::Service {
  ::grpc::Status Echo(::grpc::ServerContext* context, const ::mmdeploy::Empty* request,
                      ::mmdeploy::Reply* response) override;

  // Init Model with model file
  ::grpc::Status Init(::grpc::ServerContext* context, const ::mmdeploy::Model* request,
                      ::mmdeploy::Reply* response) override;
  // Get output names
  ::grpc::Status OutputNames(::grpc::ServerContext* context, const ::mmdeploy::Empty* request,
                             ::mmdeploy::Names* response) override;
  // Inference with inputs
  ::grpc::Status Inference(::grpc::ServerContext* context, const ::mmdeploy::TensorList* request,
                           ::mmdeploy::Reply* response) override;
  // Destroy handle
  ::grpc::Status Destroy(::grpc::ServerContext* context, const ::mmdeploy::Empty* request,
                         ::mmdeploy::Reply* response) override;

  void SaveDLC(const ::mmdeploy::Model* request, const std::string& name);

  void LoadFloatData(const std::string& data, std::vector<float>& vec);

  zdl::DlSystem::Runtime_t CheckRuntime(zdl::DlSystem::Runtime_t runtime, bool& staticQuantization);

  void Build(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
             zdl::DlSystem::Runtime_t runtime, zdl::DlSystem::RuntimeList runtimeList,
             bool useUserSuppliedBuffers, zdl::DlSystem::PlatformConfig platformConfig);

  std::string ShapeStr(zdl::DlSystem::ITensor* pTensor);

  std::string ContentStr(zdl::DlSystem::ITensor* pTensor);

  std::unique_ptr<zdl::SNPE::SNPE> snpe;
  std::unique_ptr<zdl::DlContainer::IDlContainer> container;
  std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputTensors;
  zdl::DlSystem::TensorMap inputTensorMap;
};

#endif
