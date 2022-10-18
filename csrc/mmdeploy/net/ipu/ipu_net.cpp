// Copyright (c) OpenMMLab. All rights reserved.

#include "ipu_net.h"

#include <iostream>
#include <string>
#include <vector>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

void copy_output(const model_runtime::TensorMemory& from, Tensor& to) {
  if (from.data_size_bytes != to.size()) {
    MMDEPLOY_ERROR("output tensor size not match");
    return;
  }
  int size = from.data_size_bytes;

  char* from_ptr = static_cast<char*>(from.data.get());

  char* pto = to.data<char>();
  for (int i = 0; i < size; i++) {
    // to->data[i] = typeid(*to).name()(pfrom[i]);  // hidden type conversion here
    pto[i] = *(from_ptr + i);
  }
}

void copy_input(const Tensor& from, model_runtime::TensorMemory& to) {
  if (from.size() != to.data_size_bytes) {
    MMDEPLOY_ERROR("input tensor size not match");
    return;
  }
  int size = from.size();

  char* to_ptr = static_cast<char*>(to.data.get());

  const char* pfrom = from.data<char>();
  for (int i = 0; i < size; i++) {
    // to->data[i] = typeid(*to).name()(pfrom[i]);  // hidden type conversion here
    *(to_ptr + i) = pfrom[i];
  }
}

// IPUNet::IPUNet() {}
IPUNet::~IPUNet() {}

Result<void> IPUNet::Init(const Value& args) {
  // Result<void> IPUNet::Init(const std::string& popef_path) {
  // auto& context = args["context"];
  // device_ = context["device"].get<Device>();
  // stream_ = context["stream"].get<Stream>();
  // auto name = args["name"].get<std::string>();
  // auto model = context["model"].get<Model>();

  // OUTCOME_TRY(auto config, model.GetModelConfig(name));
  // //   OUTCOME_TRY(auto onnx, model.ReadFile(config.net))
  std::string popef_path = args["popef_path"].get<std::string>();

  mconfig.device_wait_config =
      model_runtime::DeviceWaitConfig(std::chrono::seconds{600}, std::chrono::seconds{1});
  model_runner = std::make_unique<model_runtime::ModelRunner>(popef_path, mconfig);

  input_desc = model_runner->getExecuteInputs();

  output_desc = model_runner->getExecuteOutputs();

  input_memory = examples::allocateHostInputData(input_desc);
  output_memory = examples::allocateHostInputData(output_desc);

  for (int i = 0; i < input_desc.size(); i++) {
    auto desc = input_desc[i];

    input_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        DataType::kFLOAT,
        desc.shape,
        desc.name,
    });
  }

  for (int i = 0; i < output_desc.size(); i++) {
    auto desc = output_desc[i];

    output_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        DataType::kFLOAT,
        desc.shape,
        desc.name,
    });
  }
  return success();
}

Result<void> IPUNet::Deinit() { return success(); }

Result<void> IPUNet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

Result<Span<Tensor>> IPUNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> IPUNet::GetOutputTensors() { return output_tensors_; }

Result<void> IPUNet::Forward() {
  OUTCOME_TRY(stream_.Wait());

  // {
  //   // copy input to itensor buffer
  //   int count = 0;
  //   for (auto& tensor : input_tensors_) {
  //     const auto& name = tensor.desc().name;
  //     copy_input(tensor, input_memory[name]);
  //     count += 1;
  //   }
  // }

  {
    output_memory = model_runner->execute(examples::toInputMemoryView(input_memory));
    output_desc = model_runner->getExecuteOutputs();
    // if (!success) {
    //   MMDEPLOY_ERROR("IPU Inference error: {}",
    //   std::string(zdl::DlSystem::getLastErrorString())); return Status(eFail);
    // }
  }

  // {
  //   for (int i = 0; i < output_tensors_.size(); i++) {
  //     auto to_tensor = output_tensors_[i];
  //     auto name = to_tensor.desc().name;

  //     copy_output(output_memory[name], to_tensor);
  //   }

  //   //   for (const OutputValueType &name_with_memory : output_memory) {
  //   //     auto &&[name, memory] = name_with_memory;
  //   //     auto to_tensor = output_tensors_[index];
  //   //     copy_output(memory, to_tensor);

  //   //     index += 1;

  //   // }
  // }
  return success();
}

class IPUNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "IPU"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    auto p = std::make_unique<IPUNet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating IPUNet: {}", r.error().message().c_str());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, IPUNetCreator);

}  // namespace mmdeploy

int ipu_test() {
  std::cout << "inside main" << std::endl;
  mmdeploy::Value net_config{
      {"popef_path",
       "/localdata/cn-customer-engineering/qiangg/cache_poptorch/5299458688024344947.popef"}};
  // std::string popef_path(
  //     "/localdata/cn-customer-engineering/qiangg/cache_poptorch/5299458688024344947.popef");

  // auto backend("ipu");
  // auto creator = Registry<Net>::Get().GetCreator(backend);

  // cout << "get creator succeed" << endl;
  // auto net = creator->Create(net_config);
  // cout << "create net succeed" << endl;
  std::cout << "to create net" << std::endl;
  mmdeploy::IPUNet* net = new mmdeploy::IPUNet();
  net->Init(net_config);
  std::cout << "create net succeed" << std::endl;
  // auto& input_tensors = net->GetInputTensors();

  // for (auto& tensor : input_tensors) {
  //   const auto& name = tensor.desc().name;
  //   int float_size = tensor.size()/4;
  //   float* from_ptr = static_cast<float*>(tensor.data.get());
  //   for (int i=0; i<float_size; i++){
  //     from_ptr[i] = (float)rand() / RAND_MAX;
  //   }

  // }
  auto result = net->Forward();

  std::string result_str;
  if (result) {
    result_str = "true";
  } else {
    result_str = "false";
  }

  std::cout << result_str << std::endl;
  delete net;

  return 0;
}
