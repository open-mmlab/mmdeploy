// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MODEL_STATE_H
#define MMDEPLOY_MODEL_STATE_H

#define MMDEPLOY_CXX_USE_OPENCV 0

#include "mmdeploy/core/model.h"
#include "mmdeploy/pipeline.hpp"
#include "triton/backend/backend_model.h"

namespace triton::backend::mmdeploy {

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model, ModelState** state);

  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const { return output_names_; }
  const std::vector<TRITONSERVER_DataType>& input_data_types() const { return input_data_types_; }
  const std::vector<TRITONSERVER_DataType>& output_data_types() const { return output_data_types_; }

  const std::vector<std::string>& input_formats() const { return input_formats_; }

  ::mmdeploy::Pipeline CreatePipeline(TRITONSERVER_InstanceGroupKind kind, int device_id);

  const std::string& task_type() { return model_.meta().task; }

 private:
  explicit ModelState(TRITONBACKEND_Model* triton_model);

  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ::mmdeploy::framework::Model model_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<TRITONSERVER_DataType> input_data_types_;
  std::vector<TRITONSERVER_DataType> output_data_types_;
  std::vector<std::string> input_formats_;
};

}  // namespace triton::backend::mmdeploy

#endif  // MMDEPLOY_MODEL_STATE_H
