// Copyright (c) OpenMMLab. All rights reserved.

#ifndef SERVICE_RISCV_IMPL_H
#define SERVICE_RISCV_IMPL_H

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <memory>

#include "backend.h"

class InferenceServiceImpl final : public Inference::Service {
  // Init Model with model file
  Status Init(ServerContext* context, const Model* request, Reply* response) override;
  // Get output names
  Status OutputNames(ServerContext* context, const Empty* request, Names* response) override;
  // Inference with inputs
  Status Inference(ServerContext* context, const TensorList* request, Reply* response) override;
  // Destroy handle
  Status Destroy(ServerContext* context, const Empty* request, Reply* response) override;

 private:
  std::unique_ptr<Net> net_;
};

#endif
