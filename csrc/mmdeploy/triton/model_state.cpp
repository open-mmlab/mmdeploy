// Copyright (c) OpenMMLab. All rights reserved.

#include "model_state.h"

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/pipeline.hpp"

namespace triton::backend::mmdeploy {

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                         std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return {};
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), model_(JoinPath({RepositoryPath(), std::to_string(Version())})) {
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
  common::TritonJson::Value inputs;
  common::TritonJson::Value outputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

  for (size_t i = 0; i < inputs.ArraySize(); ++i) {
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));

    triton::common::TritonJson::Value reshape;
    RETURN_ERROR_IF_TRUE(input.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                         std::string("reshape not supported for input tensor"));

    std::string name;
    RETURN_IF_ERROR(input.MemberAsString("name", &name));
    input_names_.push_back(name);

    std::string data_type;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &data_type));
    input_data_types_.push_back(ModelConfigDataTypeToTritonServerDataType(data_type));

    std::string format;
    RETURN_IF_ERROR(input.MemberAsString("format", &format));
    input_formats_.push_back(format);
  }

  for (size_t i = 0; i < outputs.ArraySize(); ++i) {
    common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

    triton::common::TritonJson::Value reshape;
    RETURN_ERROR_IF_TRUE(output.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                         std::string("reshape not supported for output tensor"));

    std::string name;
    RETURN_IF_ERROR(output.MemberAsString("name", &name));
    output_names_.push_back(name);

    std::string data_type;
    RETURN_IF_ERROR(output.MemberAsString("data_type", &data_type));
    output_data_types_.push_back(ModelConfigDataTypeToTritonServerDataType(data_type));
  }

  return {};
}

::mmdeploy::Pipeline ModelState::CreatePipeline(TRITONSERVER_InstanceGroupKind kind,
                                                int device_id) {
  // infer device name
  std::string device_name = "cpu";
  if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    device_name = "cuda";
  }

  auto config = model_.ReadConfig("pipeline.json").value();

  config["context"]["model"] = model_;

  ::mmdeploy::Context context(::mmdeploy::Device(device_name, device_id));
  return {config, context};
}

}  // namespace triton::backend::mmdeploy
