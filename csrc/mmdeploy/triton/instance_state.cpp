// Copyright (c) OpenMMLab. All rights reserved.

#include "instance_state.h"

#include <numeric>

#include "convert.h"
#include "json.hpp"
#include "mmdeploy/archive/json_archive.h"
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
      pipeline_(model_state_->CreatePipeline(Kind(), DeviceId())) {}

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

  BackendInputCollector collector(requests, request_count, &responses,
                                  model_state->TritonMemoryManager(), false /* pinned_enabled */,
                                  nullptr /* stream*/);

  // To instruct ProcessTensor to "gather" the entire batch of input
  // tensors into a single contiguous buffer in CPU memory, set the
  // "allowed input types" to be the CPU ones (see tritonserver.h in
  // the triton-inference-server/core repo for allowed memory types).
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types = {
      {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

  std::vector<const char*> input_buffers(model_state->input_names().size());

  std::vector<std::unique_ptr<BackendInputCollector>> collectors(request_count);
  std::vector<std::vector<TRITONBACKEND_Response*>> response_vecs(request_count);

  ::mmdeploy::Value::Array image_and_metas_array;
  ::mmdeploy::Value::Array input_tensors_array;

  // Setting input data
  for (uint32_t request_index = 0; request_index < request_count; ++request_index) {
    ::mmdeploy::Value::Object input_tensors;
    ::mmdeploy::Value::Object image_and_metas;
    response_vecs[request_index] = {responses[request_index]};
    collectors[request_index] = std::make_unique<BackendInputCollector>(
        &requests[request_index], 1, &response_vecs[request_index],
        model_state->TritonMemoryManager(), false, nullptr);

    for (size_t input_id = 0; input_id < model_state->input_names().size(); ++input_id) {
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
        RETURN_IF_ERROR(collectors[request_index]->ProcessTensor(
            input_name.c_str(), nullptr, 0, allowed_input_types, &buffer, &buffer_size,
            &memory_type, &memory_type_id));

        ::mmdeploy::framework::Device device(0);
        if (memory_type == TRITONSERVER_MEMORY_GPU) {
          device = ::mmdeploy::framework::Device("cuda", static_cast<int>(memory_type_id));
        }
        if (model_state->input_formats()[request_index] == "FORMAT_NHWC") {
          // Construct Mat from shape & buffer
          ::mmdeploy::framework::Mat mat(
              static_cast<int>(dims[0]), static_cast<int>(dims[1]), ::mmdeploy::PixelFormat::kBGR,
              ::mmdeploy::DataType::kINT8,
              std::shared_ptr<void>(const_cast<char*>(buffer), [](auto) {}), device);
          image_and_metas.insert({input_name, mat});
        } else {
          ::mmdeploy::framework::Tensor tensor(
              ::mmdeploy::framework::TensorDesc{
                  device, ::mmdeploy::DataType::kFLOAT,
                  ::mmdeploy::framework::TensorShape(dims, dims + dims_count), input_name},
              std::shared_ptr<void>(const_cast<char*>(buffer), [](auto) {}));
          input_tensors.insert({input_name, std::move(tensor)});
        }
      } else {
        ::mmdeploy::Value value;
        GetStringInputTensor(input, dims, dims_count, value);
        assert(value.is_array());
        ::mmdeploy::update(image_and_metas, value.front().object(), 2);
      }
    }

    if (!input_tensors.empty()) {
      input_tensors_array.emplace_back(std::move(input_tensors));
    }
    if (!image_and_metas.empty()) {
      image_and_metas_array.emplace_back(std::move(image_and_metas));
    }

    // Input from device memory is not supported yet
    const bool need_cuda_input_sync = collectors[request_index]->Finalize();
    if (need_cuda_input_sync) {
#if TRITON_ENABLE_GPU
      cudaStreamSynchronize(CudaStream());
#else
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  "mmdeploy backend: unexpected CUDA sync required by collector");
#endif
    }
  }

  ::mmdeploy::Value input_args;
  if (!image_and_metas_array.empty()) {
    input_args.push_back(std::move(image_and_metas_array));
  }
  if (!input_tensors_array.empty()) {
    input_args.push_back(std::move(input_tensors_array));
  }

  MMDEPLOY_DEBUG("input: {}", input_args);

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  ::mmdeploy::Value outputs = pipeline_.Apply(input_args);

  std::vector<std::string> strings;
  auto output_tensors =
      ConvertOutputToTensors(model_state->task_type(), request_count, outputs, strings);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  std::vector<std::unique_ptr<BackendOutputResponder>> responders(request_count);
  MMDEPLOY_DEBUG("request_count {}", request_count);
  for (uint32_t request_index = 0; request_index < request_count; ++request_index) {
    responders[request_index] = std::make_unique<BackendOutputResponder>(
        &requests[request_index], 1, &response_vecs[request_index],
        model_state->TritonMemoryManager(), false, false, nullptr);
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
#endif  // TRITON_ENABLE_STATS

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
