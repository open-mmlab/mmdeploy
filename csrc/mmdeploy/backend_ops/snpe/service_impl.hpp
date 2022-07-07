#ifndef SERVICE_IMPL_H
#define SERVICE_IMPL_H

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "inference.grpc.pb.h"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"

#include "DiagLog/IDiagLog.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using mmdeploy::Inference;
using mmdeploy::Model;
using mmdeploy::Empty;
using mmdeploy::Reply;
using mmdeploy::Tensor;
using mmdeploy::TensorList;

// Logic and data behind the server's behavior.
class InferenceServiceImpl final : public Inference::Service {
    ::grpc::Status Echo(::grpc::ServerContext* context, const ::mmdeploy::Empty* request, ::mmdeploy::Reply* response) override;

    // Init Model with model file
    ::grpc::Status Init(::grpc::ServerContext* context, const ::mmdeploy::Model* request, ::mmdeploy::Reply* response) override;
    // Get output names
    ::grpc::Status OutputNames(::grpc::ServerContext* context, const ::mmdeploy::Empty* request, ::mmdeploy::Names* response) override;
    // Inference with inputs 
    ::grpc::Status Inference(::grpc::ServerContext* context, const ::mmdeploy::TensorList* request, ::mmdeploy::Reply* response) override;
    // Destory handle
    ::grpc::Status Destory(::grpc::ServerContext* context, const ::mmdeploy::Empty* request, ::mmdeploy::Reply* response) override;

    std::string save_dlc(const ::mmdeploy::Model* request);

    void load_float_data(const std::string& data, std::vector<float>& vec);

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
};

#endif
