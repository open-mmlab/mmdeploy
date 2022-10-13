// Copyright (c) OpenMMLab. All rights reserved.

#include "ipu_net.h"

#include <string>
#include <vector>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

void copy_output(const model_runtime::TensorMemory& from, Tensor& to) {
  if (from.size() != to->data_size_bytes) {
    MMDEPLOY_ERROR("output tensor size not match");
    return;
  }
  int size = from.size();

  const float* pfrom = from.data<float>();
  for (int i = 0; i < size; i++) {
    // to->data[i] = typeid(*to).name()(pfrom[i]);  // hidden type conversion here
    to.data[i] = pfrom[i];
  }
}

void copy_input(const Tensor& from, model_runtime::TensorMemory& to) {
  if (from.size() != to->data_size_bytes) {
    MMDEPLOY_ERROR("input tensor size not match");
    return;
  }
  int size = from.size();

  const float* pfrom = from.data<float>();
  for (int i = 0; i < size; i++) {
    // to->data[i] = typeid(*to).name()(pfrom[i]);  // hidden type conversion here
    to.data[i] = pfrom[i];
  }
}

IPUNet::~IPUNet() {}

Result<void> IPUNet::Init(const Value& args) {
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();

  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  //   OUTCOME_TRY(auto onnx, model.ReadFile(config.net))

  config.device_wait_config = model_runtime::DeviceWaitConfig{600s /*timeout*/, 1s /*sleep_time*/};
  model_runner(config.net, config);

  input_desc = model_runner.getExecuteInputs();

  output_desc = model_runner.getExecuteOutputs();

  input_memory = examples::allocateHostInputData(input_desc);

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

  {
    // copy input to itensor buffer
    for (auto& tensor : input_tensors_) {
      const auto& name = tensor.desc().name;
      TensorMemory input_ = input_memory[name];
      copy_input(tensor, input_);
    }
  }

  {
    output_memory = model_runner.execute(examples::toInputMemoryView(input_memory));
    output_desc = model_runner.getExecuteOutputs();
    // if (!success) {
    //   MMDEPLOY_ERROR("IPU Inference error: {}",
    //   std::string(zdl::DlSystem::getLastErrorString())); return Status(eFail);
    // }
  }

  {
    for (int i = 0; i < output_tensors.size(); i++) {
      auto to_tensor = output_tensors_[i];
      auto name = to_tensor.desc().name;
      auto memory = output_memory[name];
      copy_output(memory, to_tensor);
    }

    //   for (const OutputValueType &name_with_memory : output_memory) {
    //     auto &&[name, memory] = name_with_memory;
    //     auto to_tensor = output_tensors_[index];
    //     copy_output(memory, to_tensor);

    //     index += 1;

    // }
  }
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
