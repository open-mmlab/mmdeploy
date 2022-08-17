// Copyright (c) OpenMMLab. All rights reserved.

#include "server_impl.h"

Status InferenceServiceImpl::Init(ServerContext* context, const Model* request, Reply* response) {
  if (net_ != nullptr) {
    net_.reset();
  }

  if (request->type() == ModelType::NCNN) {
    net_ = std::make_unique<NCNNNet>();
    return net_->Init(context, request, response);
  } else if (request->type() == ModelType::PPLNN) {
    response->set_info("not implemented");
    response->set_status(-1);
  } else {
    response->set_info("unsupported model type");
    response->set_status(-1);
  }

  return Status::OK;
}

Status InferenceServiceImpl::OutputNames(ServerContext* context, const Empty* request,
                                         Names* response) {
  return net_->OutputNames(context, request, response);
}

Status InferenceServiceImpl::Inference(ServerContext* context, const TensorList* request,
                                       Reply* response) {
  return net_->Inference(context, request, response);
}
Status InferenceServiceImpl::Destroy(ServerContext* context, const Empty* request,
                                     Reply* response) {
  net_.reset();
  response->set_status(0);
  return Status::OK;
}
