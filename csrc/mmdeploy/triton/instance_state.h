// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_INSTANCE_STATE_H
#define MMDEPLOY_INSTANCE_STATE_H

#include "mmdeploy/core/tensor.h"
#include "model_state.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

namespace triton::backend::mmdeploy {

class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(ModelState* model_state,
                                    TRITONBACKEND_ModelInstance* triton_model_instance,
                                    ModelInstanceState** state);
  ~ModelInstanceState() override = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  TRITONSERVER_Error* Execute(TRITONBACKEND_Request** requests, uint32_t request_count);

  TRITONSERVER_Error* GetStringInputTensor(TRITONBACKEND_Input* input, const int64_t* dims,
                                           uint32_t dims_count, ::mmdeploy::Value& value);

  TRITONSERVER_Error* SetStringOutputTensor(const ::mmdeploy::framework::Tensor& tensor,
                                            const std::vector<std::string>& strings,
                                            TRITONBACKEND_Response* response);

 private:
  ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance);

 private:
  ModelState* model_state_;
  ::mmdeploy::Pipeline pipeline_;
};

}  // namespace triton::backend::mmdeploy

#endif  // MMDEPLOY_INSTANCE_STATE_H
