#ifndef SERVER_RISCV_BACKEND_H
#define SERVER_RISCV_BACKEND_H

#include "inference.grpc.pb.h"
// It's ncnn's net.h
#include "net.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using mmdeploy::Empty;
using mmdeploy::Inference;
using mmdeploy::Model;
using mmdeploy::ModelType;
using mmdeploy::Names;
using mmdeploy::Reply;
using mmdeploy::Tensor;
using mmdeploy::TensorList;

class Net {
 public:
  virtual Status Init(ServerContext* context, const Model* request, Reply* response) = 0;
  virtual Status OutputNames(ServerContext* context, const Empty* request, Names* response) = 0;
  virtual Status Inference(ServerContext* context, const TensorList* request, Reply* response) = 0;
};

class NCNNNet : public Net {
 public:
  Status Init(ServerContext* context, const Model* request, Reply* response) override;
  Status OutputNames(ServerContext* context, const Empty* request, Names* response) override;
  Status Inference(ServerContext* context, const TensorList* request, Reply* response) override;

 private:
  ncnn::Net net_;
  std::string params_;
  std::string weights_;
};

#endif
