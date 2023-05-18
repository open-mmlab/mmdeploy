// Copyright (c) OpenMMLab. All rights reserved.

#include "instance_state.h"

#include <numeric>
#include <sstream>

#include "convert.h"
#include "json.hpp"
#include "json_input.h"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy_utils.h"

namespace triton::backend::mmdeploy {

TRITONSERVER_Error* ModelInstanceState::Create(ModelState* model_state,
                                               TRITONBACKEND_ModelInstance* triton_model_instance,
                                               ModelInstanceState** state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                         std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}
ModelInstanceState::ModelInstanceState(ModelState* model_state,
                                       TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state),
      pipeline_(model_state_->CreatePipeline(Kind(), DeviceId())) {
  // parse parameters
  ::triton::common::TritonJson::Value parameters;
  model_state->ModelConfig().Find("parameters", &parameters);
  std::string info;
  TryParseModelStringParameter(parameters, "merge_inputs", &info, "");
  if (info != "") {
    std::stringstream ss1(info);
    std::string group;
    while (std::getline(ss1, group, ',')) {
      std::stringstream ss2(group);
      merge_inputs_.emplace_back();
      int v;
      while (ss2 >> v) {
        merge_inputs_.back().push_back(v);
      }
    }
  }
}

//     TRITON               DIR      MMDeploy
// (Tensor, PixFmt, Region) ->  (Mat     , Region)
// [Tensor]                 <-  ([Tensor], Meta  )
// [Tensor]                 ->  ([Tensor], Meta  )
// [Tensor]                 <-  [Value]

TRITONSERVER_Error* ModelInstanceState::Execute(TRITONBACKEND_Request** requests,
                                                uint32_t request_count) {
  // Collect various timestamps during the execution of this batch or
  // requests. These values are reported below before returning from
  // the function.

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  ModelState* model_state = StateForModel();

  const int max_batch_size = model_state->MaxBatchSize();

  for (size_t i = 0; i < request_count; ++i) {
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string("null request given to MMDeploy backend for '" + Name() + "'").c_str()));
      return nullptr;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      // and batch dim should be 1 for mmdeploy
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr, nullptr,
                                            nullptr);
        if (err == nullptr && shape[0] != 1) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string("only support batch dim 1 for single request").c_str());
        }
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return nullptr;
      }
    }
  }

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types = {
      {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

  std::vector<std::unique_ptr<BackendInputCollector>> collectors(request_count);
  std::vector<std::vector<TRITONBACKEND_Response*>> response_vecs(request_count);
  bool need_cuda_input_sync = false;

  for (uint32_t request_index = 0; request_index < request_count; ++request_index) {
    response_vecs[request_index] = {responses[request_index]};
    collectors[request_index] = std::make_unique<BackendInputCollector>(
        &requests[request_index], 1, &response_vecs[request_index],
        model_state->TritonMemoryManager(), false, CudaStream());
  }

  // Setting input data
  ::mmdeploy::Value vec_inputs;
  std::vector<int> batch_per_request;
  for (uint32_t request_index = 0; request_index < request_count; ++request_index) {
    const auto& collector = collectors[request_index];
    ::mmdeploy::Value vec_inputi;
    batch_per_request.push_back(1);

    for (size_t input_id = 0; input_id < model_state->input_names().size(); ++input_id) {
      ::mmdeploy::Value inputi;
      const auto& input_name = model_state->input_names()[input_id];
      // Get input shape
      TRITONBACKEND_Input* input{};
      RETURN_IF_ERROR(
          TRITONBACKEND_RequestInput(requests[request_index], input_name.c_str(), &input));
      TRITONSERVER_DataType data_type{};
      const int64_t* dims{};
      uint32_t dims_count{};
      RETURN_IF_ERROR(TRITONBACKEND_InputProperties(input, nullptr, &data_type, &dims, &dims_count,
                                                    nullptr, nullptr));
      if (data_type != TRITONSERVER_TYPE_BYTES) {
        // Collect input buffer
        const char* buffer{};
        size_t buffer_size{};
        TRITONSERVER_MemoryType memory_type{};
        int64_t memory_type_id{};
        RETURN_IF_ERROR(collector->ProcessTensor(input_name.c_str(), nullptr, 0,
                                                 allowed_input_types, &buffer, &buffer_size,
                                                 &memory_type, &memory_type_id));
        ::mmdeploy::framework::Device device(0);
        if (memory_type == TRITONSERVER_MEMORY_GPU) {
          device = ::mmdeploy::framework::Device("cuda", static_cast<int>(memory_type_id));
        }

        if (model_state->input_formats()[input_id] == "FORMAT_NHWC") {
          // Construct Mat from shape & buffer
          int h, w;
          if (max_batch_size > 0) {
            h = dims[1];
            w = dims[2];
          } else {
            h = dims[0];
            w = dims[1];
          }
          ::mmdeploy::framework::Mat mat(
              h, w, ::mmdeploy::PixelFormat::kBGR, ::mmdeploy::DataType::kINT8,
              std::shared_ptr<void>(const_cast<char*>(buffer), [](auto) {}), device);
          inputi = {{input_name, mat}};
        } else {
          ::mmdeploy::framework::Tensor tensor(
              ::mmdeploy::framework::TensorDesc{
                  device, ConvertDataType(model_state->input_data_types()[input_id]),
                  ::mmdeploy::framework::TensorShape(dims, dims + dims_count), input_name},
              std::shared_ptr<void>(const_cast<char*>(buffer), [](auto) {}));
          inputi = {{input_name, std::move(tensor)}};
        }
      } else {
        ::mmdeploy::Value value;
        GetStringInputTensor(input, dims, dims_count, value);
        assert(value.is_array());

        if (value[0].contains("type")) {
          const auto& type = value[0]["type"].get_ref<std::string&>();
          CreateJsonInput(value[0]["value"], type, inputi);
          batch_per_request.back() = inputi.size();
        } else {
          inputi = {{}};
          inputi.update(value.front().object());
        }
      }
      vec_inputi.push_back(std::move(inputi));  // [ a, [b,b] ]
    }

    // broadcast, [ a, [b,b] ] -> [[a, a], [b, b]]
    if (batch_per_request.back() >= 1) {
      // std::vector<::mmdeploy::Value> input;
      ::mmdeploy::Value input;
      for (size_t i = 0; i < vec_inputi.size(); i++) {
        input.push_back(::mmdeploy::Value::kArray);
      }

      for (int i = 0; i < batch_per_request.back(); i++) {
        for (size_t input_id = 0; input_id < model_state->input_names().size(); ++input_id) {
          if (vec_inputi[input_id].is_object()) {
            input[input_id].push_back(vec_inputi[input_id]);
          } else {
            input[input_id].push_back(vec_inputi[input_id][i]);
          }
        }
      }
      vec_inputi = input;
    }

    // construct [[a,a,a], [b,b,b]]
    if (vec_inputs.is_null()) {
      for (size_t i = 0; i < vec_inputi.size(); i++) {
        vec_inputs.push_back(::mmdeploy::Value::kArray);
      }
    }
    for (size_t i = 0; i < vec_inputi.size(); i++) {
      auto&& inner = vec_inputi[i];
      for (auto&& obj : inner) {
        vec_inputs[i].push_back(std::move(obj));
      }
    }
  }

  // merge inputs for example: [[a,a,a], [b,b,b], [c,c,c]] -> [[aaa], [(b,c), (b,c), (b,c)]]
  if (!merge_inputs_.empty()) {
    int n_example = vec_inputs[0].size();
    ::mmdeploy::Value inputs;
    for (const auto& group : merge_inputs_) {
      ::mmdeploy::Value input_array;
      for (int i = 0; i < n_example; i++) {
        ::mmdeploy::Value input_i;
        for (const auto& idx : group) {
          auto&& inner = vec_inputs[idx];
          input_i.update(inner[i]);
        }
        input_array.push_back(std::move(input_i));
      }
      inputs.push_back(std::move(input_array));
    }
    vec_inputs = std::move(inputs);
  }

  if (need_cuda_input_sync) {
#if TRITON_ENABLE_GPU
    cudaStreamSynchronize(CudaStream());
#else
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                "mmdeploy backend: unexpected CUDA sync required by collector");
#endif
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  ::mmdeploy::Value outputs = pipeline_.Apply(vec_inputs);
  // MMDEPLOY_ERROR("outputs:\n{}", outputs);

  // preprocess and inference need cuda sync
  {
    std::string device_name = "cpu";
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      device_name = "cuda";
    }
    auto device = ::mmdeploy::framework::Device(device_name.c_str(), DeviceId());
    auto stream = ::mmdeploy::framework::Stream::GetDefault(device);
    stream.Wait();
  }

  std::vector<std::string> strings;
  auto output_tensors = ConvertOutputToTensors(model_state->task_type(), request_count,
                                               batch_per_request, outputs, strings);
  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  std::vector<std::unique_ptr<BackendOutputResponder>> responders(request_count);
  MMDEPLOY_DEBUG("request_count {}", request_count);
  for (uint32_t request_index = 0; request_index < request_count; ++request_index) {
    responders[request_index] = std::make_unique<BackendOutputResponder>(
        &requests[request_index], 1, &response_vecs[request_index], false,
        model_state->TritonMemoryManager(), false, CudaStream());
    for (size_t output_id = 0; output_id < model_state->output_names().size(); ++output_id) {
      auto output_name = model_state->output_names()[output_id];
      MMDEPLOY_DEBUG("output name {}", output_name);
      auto output_data_type = model_state->output_data_types()[output_id];
      for (const auto& tensor : output_tensors[request_index]) {
        if (tensor.name() == output_name) {
          if (output_data_type != TRITONSERVER_TYPE_BYTES) {
            auto shape = tensor.shape();
            MMDEPLOY_DEBUG("name {}, shape {}", tensor.name(), shape);
            auto memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t memory_type_id = 0;
            if (not tensor.device().is_host()) {
              memory_type = TRITONSERVER_MEMORY_GPU;
              memory_type_id = tensor.device().device_id();
            }
            responders[request_index]->ProcessTensor(
                tensor.name(), ConvertDataType(tensor.data_type()), shape, tensor.data<char>(),
                memory_type, memory_type_id);
          } else {
            RETURN_IF_ERROR(SetStringOutputTensor(tensor, strings, responses[request_index]));
          }
          break;
        }
      }
    }

    const bool need_cuda_output_sync = responders[request_index]->Finalize();
    if (need_cuda_output_sync) {
#if TRITON_ENABLE_GPU
      cudaStreamSynchronize(CudaStream());
#else
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  "mmdeploy backend: unexpected CUDA sync required by responder");
#endif
    }
  }

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
  // For batch statistics need to know the total batch size of the
  // requests. This is not necessarily just the number of requests,
  // because if the model supports batching then any request can be a
  // batched request itself.
  size_t total_batch_size = request_count;
#else
  (void)exec_start_ns;
  (void)exec_end_ns;
  (void)compute_start_ns;
  (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

  // Report statistics for each request, and then release the request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];

#ifdef TRITON_ENABLE_STATS
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
                     TritonModelInstance(), request, (responses[r] != nullptr) /* success */,
                     exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
                 "failed reporting request statistics");
#endif  // TRITON_ENABLE_STATS

    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
                 "failed releasing request");
  }

#ifdef TRITON_ENABLE_STATS
  // Report batch statistics.
  LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                   TritonModelInstance(), total_batch_size, exec_start_ns, compute_start_ns,
                   compute_end_ns, exec_end_ns),
               "failed reporting batch request statistics");
#endif             // TRITON_ENABLE_STATS

  return nullptr;  // success
}

TRITONSERVER_Error* ModelInstanceState::GetStringInputTensor(TRITONBACKEND_Input* input,
                                                             const int64_t* dims,
                                                             uint32_t dims_count,
                                                             ::mmdeploy::Value& value) {
  ::mmdeploy::Value::Array array;
  const char* buffer{};
  uint64_t buffer_byte_size{};
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id{};
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(input, 0, reinterpret_cast<const void**>(&buffer),
                                            &buffer_byte_size, &memory_type, &memory_type_id));
  auto count = std::accumulate(dims, dims + dims_count, 1LL, std::multiplies<>{});
  size_t offset = 0;
  for (int64_t i = 0; i < count; ++i) {
    // read string length
    if (offset + sizeof(uint32_t) > buffer_byte_size) {
      break;
    }
    auto length = *reinterpret_cast<const uint32_t*>(buffer + offset);
    offset += sizeof(uint32_t);
    // read string data
    if (offset + length > buffer_byte_size) {
      break;
    }
    std::string data(buffer + offset, buffer + offset + length);
    offset += length;
    // deserialize from json string
    auto data_value = ::mmdeploy::from_json<::mmdeploy::Value>(nlohmann::json::parse(data));
    array.push_back(std::move(data_value));
  }
  value = std::move(array);
  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::SetStringOutputTensor(
    const ::mmdeploy::framework::Tensor& tensor, const std::vector<std::string>& strings,
    TRITONBACKEND_Response* response) {
  assert(tensor.data_type() == ::mmdeploy::DataType::kINT32);
  TRITONSERVER_Error* err{};
  TRITONBACKEND_Output* response_output{};
  err = TRITONBACKEND_ResponseOutput(response, &response_output, tensor.name(),
                                     TRITONSERVER_TYPE_BYTES, tensor.shape().data(),
                                     tensor.shape().size());
  if (!err) {
    size_t data_byte_size{};
    auto index_data = tensor.data<int32_t>();
    auto size = tensor.size();
    for (int64_t j = 0; j < size; ++j) {
      data_byte_size += strings[index_data[j]].size();
    }
    auto expected_byte_size = data_byte_size + sizeof(uint32_t) * size;
    void* buffer{};
    TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t actual_memory_type_id = 0;
    err = TRITONBACKEND_OutputBuffer(response_output, &buffer, expected_byte_size,
                                     &actual_memory_type, &actual_memory_type_id);
    if (!err) {
      bool cuda_used = false;
      size_t copied_byte_size = 0;
      for (int64_t j = 0; j < size; ++j) {
        auto len = static_cast<uint32_t>(strings[index_data[j]].size());
        err = CopyBuffer(tensor.name(), TRITONSERVER_MEMORY_CPU, 0, actual_memory_type,
                         actual_memory_type_id, sizeof(uint32_t), &len,
                         static_cast<char*>(buffer) + copied_byte_size, nullptr, &cuda_used);
        if (err) {
          break;
        }
        copied_byte_size += sizeof(uint32_t);
        err = CopyBuffer(tensor.name(), TRITONSERVER_MEMORY_CPU, 0, actual_memory_type,
                         actual_memory_type_id, len, strings[index_data[j]].data(),
                         static_cast<char*>(buffer) + copied_byte_size, nullptr, &cuda_used);
        if (err) {
          break;
        }
        copied_byte_size += len;
      }
    }
  }
  return err;
}

}  // namespace triton::backend::mmdeploy
