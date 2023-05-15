// Copyright (c) OpenMMLab. All rights reserved.

#include "model_state.h"

#include <fstream>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/filesystem.h"
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

ModelState::ModelState(TRITONBACKEND_Model* triton_model) : BackendModel(triton_model) {
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

  std::string pipeline_template_path =
      JoinPath({RepositoryPath(), std::to_string(Version()), "pipeline_template.json"});
  if (fs::exists(pipeline_template_path)) {
    std::ifstream ifs(pipeline_template_path, std::ios::binary | std::ios::in);
    ifs.seekg(0, std::ios::end);
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string str(size, '\0');
    ifs.read(str.data(), size);

    auto config = ::mmdeploy::from_json<::mmdeploy::Value>(nlohmann::json::parse(str));
    ::mmdeploy::Context context(::mmdeploy::Device(device_name, device_id));
    config["task_type"].get_to(task_type_);
    config.object().erase("task_type");
    if (config.contains("model_names")) {
      std::vector<std::string> model_names;
      ::mmdeploy::from_value(config["model_names"], model_names);
      for (const auto& name : model_names) {
        std::string model_path = JoinPath({RepositoryPath(), std::to_string(Version()), name});
        context.Add(name, ::mmdeploy::Model(model_path));
      }
      config.object().erase("model_names");
    }
    return {config, context};

  } else {
    model_ = ::mmdeploy::framework::Model(JoinPath({RepositoryPath(), std::to_string(Version())}));
    auto config = model_.ReadConfig("pipeline.json").value();
    config["context"]["model"] = model_;
    ::mmdeploy::Context context(::mmdeploy::Device(device_name, device_id));
    task_type_ = model_.meta().task;
    return {config, context};
  }
}

}  // namespace triton::backend::mmdeploy
