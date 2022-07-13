// Copyright (c) OpenMMLab. All rights reserved.

#include "service_impl.h"

#include <getopt.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

zdl::DlSystem::Runtime_t InferenceServiceImpl::CheckRuntime(zdl::DlSystem::Runtime_t runtime,
                                                            bool& staticQuantization) {
  static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

  fprintf(stdout, "SNPE Version: %s\n", Version.asString().c_str());

  if ((runtime != zdl::DlSystem::Runtime_t::DSP) && staticQuantization) {
    fprintf(stderr,
            "ERROR: Cannot use static quantization with CPU/GPU runtimes. "
            "It is only designed for DSP/AIP runtimes.\n"
            "ERROR: Proceeding without static quantization on selected "
            "runtime.\n");
    staticQuantization = false;
  }

  if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
    fprintf(stderr, "Selected runtime not present. Falling back to CPU.\n");
    runtime = zdl::DlSystem::Runtime_t::CPU;
  }

  return runtime;
}

void InferenceServiceImpl::Build(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                 zdl::DlSystem::Runtime_t runtime,
                                 zdl::DlSystem::RuntimeList runtimeList,
                                 bool useUserSuppliedBuffers,
                                 zdl::DlSystem::PlatformConfig platformConfig) {
  zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

  if (runtimeList.empty()) {
    runtimeList.add(runtime);
  }

  snpe = snpeBuilder.setOutputLayers({})
             .setRuntimeProcessorOrder(runtimeList)
             .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
             .setPlatformConfig(platformConfig)
             .setExecutionPriorityHint(zdl::DlSystem::ExecutionPriorityHint_t::HIGH)
             .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE)
             .build();
  return;
}

void InferenceServiceImpl::SaveDLC(const ::mmdeploy::Model* request, const std::string& filename) {
  auto model = request->weights();
  fprintf(stdout, "saving file to %s\n", filename.c_str());
  std::ofstream fout;
  fout.open(filename, std::ios::binary | std::ios::out);
  fout.write(model.data(), model.size());
  fout.flush();
  fout.close();
}

void InferenceServiceImpl::LoadFloatData(const std::string& data, std::vector<float>& vec) {
  size_t len = data.size();
  assert(len % sizeof(float) == 0);
  const char* ptr = data.data();
  for (int i = 0; i < len; i += sizeof(float)) {
    vec.push_back(*(float*)(ptr + i));
  }
}

::grpc::Status InferenceServiceImpl::Echo(::grpc::ServerContext* context,
                                          const ::mmdeploy::Empty* request,
                                          ::mmdeploy::Reply* response) {
  fprintf(stdout, "Stage Echo: recv command\n");
  response->set_info("echo");
  return Status::OK;
}

// Logic and data behind the server's behavior.
::grpc::Status InferenceServiceImpl::Init(::grpc::ServerContext* context,
                                          const ::mmdeploy::Model* request,
                                          ::mmdeploy::Reply* response) {
  zdl::SNPE::SNPEFactory::initializeLogging(zdl::DlSystem::LogLevel_t::LOG_ERROR);
  zdl::SNPE::SNPEFactory::setLogLevel(zdl::DlSystem::LogLevel_t::LOG_ERROR);

  fprintf(stdout, "Stage Init: recv command\n");

  const std::string filename = "end2end.dlc";
  SaveDLC(request, filename);

  if (snpe != nullptr) {
    snpe.reset();
  }
  if (container != nullptr) {
    container.reset();
  }

  container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(filename));
  if (container == nullptr) {
    fprintf(stdout, "Stage Init: load dlc failed.\n");

    response->set_status(-1);
    response->set_info(zdl::DlSystem::getLastErrorString());
    return Status::OK;
  }

  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::GPU;
  if (request->has_device()) {
    switch (request->device()) {
      case mmdeploy::Model_Device_GPU:
        runtime = zdl::DlSystem::Runtime_t::GPU;
        break;
      case mmdeploy::Model_Device_DSP:
        runtime = zdl::DlSystem::Runtime_t::DSP;
      default:
        break;
    }
  }

  if (runtime != zdl::DlSystem::Runtime_t::CPU) {
    bool static_quant = false;
    runtime = CheckRuntime(runtime, static_quant);
  }

  zdl::DlSystem::RuntimeList runtimeList;
  runtimeList.add(runtime);
  zdl::DlSystem::PlatformConfig platformConfig;

  {
    ScopeTimer timer("build snpe");
    Build(container, runtime, runtimeList, false, platformConfig);
  }

  if (snpe == nullptr) {
    response->set_status(-2);
    response->set_info(zdl::DlSystem::getLastErrorString());
  }

  // setup logger
  auto logger_opt = snpe->getDiagLogInterface();
  if (!logger_opt) throw std::runtime_error("SNPE failed to obtain logging interface");
  auto logger = *logger_opt;
  auto opts = logger->getOptions();
  static std::string OutputDir = "./output/";

  opts.LogFileDirectory = OutputDir;
  if (!logger->setOptions(opts)) {
    std::cerr << "Failed to set options" << std::endl;
    return Status::OK;
  }
  if (!logger->start()) {
    std::cerr << "Failed to start logger" << std::endl;
    return Status::OK;
  }

  const auto& inputTensorNamesRef = snpe->getInputTensorNames();
  const auto& inputTensorNames = *inputTensorNamesRef;

  inputTensors.resize(inputTensorNames.size());
  for (int i = 0; i < inputTensorNames.size(); ++i) {
    const char* pname = inputTensorNames.at(i);
    const auto& shape_opt = snpe->getInputDimensions(pname);
    const auto& shape = *shape_opt;

    fprintf(stdout, "Stage Init: input tensor info:\n");
    switch (shape.rank()) {
      case 1:
        fprintf(stdout, "name: %s, shape: [%ld]\n", pname, shape[0]);
        break;
      case 2:
        fprintf(stdout, "name: %s, shape: [%ld,%ld]\n", pname, shape[0], shape[1]);
        break;
      case 3:
        fprintf(stdout, "name: %s, shape: [%ld,%ld,%ld]\n", pname, shape[0], shape[1], shape[2]);
        break;
      case 4:
        fprintf(stdout, "name: %s, shape: [%ld,%ld,%ld,%ld]\n", pname, shape[0], shape[1], shape[2],
                shape[3]);
        break;
    }
    inputTensors[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(shape);
    inputTensorMap.add(pname, inputTensors[i].get());
  }

  response->set_status(0);
  response->set_info("Stage Init: success");
  return Status::OK;
}

void InferenceServiceImpl::PrintTensorInfo(const char* pname, zdl::DlSystem::ITensor* pTensor) {
  auto shape = pTensor->getShape();
  switch (shape.rank()) {
    case 1:
      fprintf(stdout, "name: %s, shape: [%ld]\n", pname, shape[0]);
      break;
    case 2:
      fprintf(stdout, "name: %s, shape: [%ld,%ld]\n", pname, shape[0], shape[1]);
      break;
    case 3:
      fprintf(stdout, "name: %s, shape: [%ld,%ld,%ld]\n", pname, shape[0], shape[1], shape[2]);
      break;
    case 4:
      fprintf(stdout, "name: %s, shape: [%ld,%ld,%ld,%ld]\n", pname, shape[0], shape[1], shape[2],
              shape[3]);
      break;
  }

  const size_t N = std::min(10UL, pTensor->getSize());
  fprintf(stdout, "\tfirst %ld value: ", N);
  auto it = pTensor->cbegin();
  for (int i = 0; i < N; ++i) {
    fprintf(stdout, "%f ", *(it + i));
  }
  fprintf(stdout, "..%f\n", *(it + pTensor->getSize() - 1));
}

::grpc::Status InferenceServiceImpl::OutputNames(::grpc::ServerContext* context,
                                                 const ::mmdeploy::Empty* request,
                                                 ::mmdeploy::Names* response) {
  const auto& outputTensorNamesRef = snpe->getOutputTensorNames();
  const auto& outputTensorNames = *outputTensorNamesRef;

  for (int i = 0; i < outputTensorNames.size(); ++i) {
    response->add_names(outputTensorNames.at(i));
  }

  return Status::OK;
}

::grpc::Status InferenceServiceImpl::Inference(::grpc::ServerContext* context,
                                               const ::mmdeploy::TensorList* request,
                                               ::mmdeploy::Reply* response) {
  // Get input names and number
  fprintf(stdout, "Stage Inference: command\n");

  const auto& inputTensorNamesRef = snpe->getInputTensorNames();

  if (!inputTensorNamesRef) {
    response->set_status(-1);
    response->set_info(zdl::DlSystem::getLastErrorString());
    return Status::OK;
  }

  const auto& inputTensorNames = *inputTensorNamesRef;
  if (inputTensorNames.size() != request->data_size()) {
    response->set_status(-2);
    response->set_info("Stage Inference: input names count not match !");
    return Status::OK;
  }

  // Load input/output buffers with TensorMap
  {
    ScopeTimer timer("convert input");

    for (int i = 0; i < request->data_size(); ++i) {
      auto tensor = request->data(i);
      std::vector<float> float_input;
      LoadFloatData(tensor.data(), float_input);

      fprintf(stdout, "Stage Inference: tensor name: %s  input data len %lu\n",
              inputTensorNames.at(i), float_input.size());

      zdl::DlSystem::ITensor* ptensor = inputTensorMap.getTensor(tensor.name().c_str());
      if (ptensor == nullptr) {
        fprintf(stderr, "Stage Inference: cannot find name: %s in input tensor map\n",
                tensor.name().c_str());
        response->set_status(-3);
        response->set_info("cannot find name in input tensor map.");
        return Status::OK;
      }

      std::copy(float_input.begin(), float_input.end(), ptensor->begin());

      PrintTensorInfo(tensor.name().c_str(), ptensor);
    }
  }

  // A tensor map for SNPE execution outputs
  zdl::DlSystem::TensorMap outputTensorMap;
  // Execute the multiple input tensorMap on the model with SNPE
  bool success = false;
  {
    ScopeTimer timer("execute");
    success = snpe->execute(inputTensorMap, outputTensorMap);

    if (!success) {
      response->set_status(-3);
      response->set_info(zdl::DlSystem::getLastErrorString());
      return Status::OK;
    }
  }

  {
    ScopeTimer timer("convert output");
    auto out_names = outputTensorMap.getTensorNames();
    for (size_t i = 0; i < out_names.size(); ++i) {
      const char* name = out_names.at(i);
      zdl::DlSystem::ITensor* pTensor = outputTensorMap.getTensor(name);
      PrintTensorInfo(name, pTensor);

      size_t data_length = pTensor->getSize();

      std::string result;
      result.resize(sizeof(float) * data_length);
      int j = 0;
      for (auto it = pTensor->cbegin(); it != pTensor->cend(); ++it, j += sizeof(float)) {
        float f = *it;
        memcpy(&result[0] + j, reinterpret_cast<char*>(&f), sizeof(float));
      }

      auto shape = pTensor->getShape();

      ::mmdeploy::Tensor* pData = response->add_data();
      pData->set_dtype("float32");
      pData->set_name(name);
      pData->set_data(result);
      for (int j = 0; j < shape.rank(); ++j) {
        pData->add_shape(shape[j]);
      }
    }
  }

  // build output status
  response->set_status(0);
  response->set_info("Stage Inference: success");
  return Status::OK;
}

::grpc::Status InferenceServiceImpl::Destroy(::grpc::ServerContext* context,
                                             const ::mmdeploy::Empty* request,
                                             ::mmdeploy::Reply* response) {
  // zdl::SNPE::SNPEFactory::terminateLogging();
  snpe.reset();
  container.reset();
  inputTensors.clear();
  response->set_status(0);
  return Status::OK;
}
