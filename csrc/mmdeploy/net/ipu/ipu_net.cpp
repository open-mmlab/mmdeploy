// Copyright (c) OpenMMLab. All rights reserved.

#include "ipu_net.h"

#include <sys/time.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <vector>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy::framework {

mmdeploy::DataType IPUNet::ipu_type_convert(const popef::DataType& ipu_type) {
  mmdeploy::DataType mtype;
  if (ipu_type == popef::DataType::F32) {
    mtype = mmdeploy::DataType::kFLOAT;
  } else if (ipu_type == popef::DataType::F16) {
    mtype = mmdeploy::DataType::kHALF;
  } else if (ipu_type == popef::DataType::F8) {
    mtype = mmdeploy::DataType::kINT8;
  } else {
    throw std::invalid_argument(
        "invalid data type for IPU backend, current legit is : fp32, fp16, fp8");
  }
  return mtype;
}

void IPUNet::copy_output(const model_runtime::TensorMemory& from, Tensor& to) {
  if (from.data_size_bytes != to.byte_size()) {
    MMDEPLOY_ERROR("output tensor size not match from size {} to size {}", from.data_size_bytes,
                   to.byte_size());
    return;
  }
  int size = from.data_size_bytes;

  MMDEPLOY_INFO("copy output total byte_size {}", size);

  char* from_ptr = static_cast<char*>(from.data.get());

  char* pto = to.data<char>();
  for (int i = 0; i < size; i++) {
    // to->data[i] = typeid(*to).name()(pfrom[i]);  // hidden type conversion here
    pto[i] = *(from_ptr + i);
  }
}

void IPUNet::copy_input(const Tensor& from, model_runtime::TensorMemory& to) {
  if (from.byte_size() != to.data_size_bytes) {
    MMDEPLOY_ERROR("input tensor size not match  from size {} to size {} ", from.byte_size(),
                   to.data_size_bytes);
    return;
  }
  int size = from.byte_size();

  MMDEPLOY_INFO("copy input total byte_size {}", size);

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

  batch_per_step = args["bps"].get<int>();

  mconfig.device_wait_config =
      model_runtime::DeviceWaitConfig(std::chrono::seconds{600}, std::chrono::seconds{1});
  model_runner = std::make_unique<model_runtime::ModelRunner>(popef_path, mconfig);

  input_desc = model_runner->getExecuteInputs();
  output_desc = model_runner->getExecuteOutputs();

  for (int i=0; i<input_desc.size(); i++){
    input_desc[i].shape[0] *= batch_per_step;
    input_desc[i].size_in_bytes *= batch_per_step;
    MMDEPLOY_INFO("input desc shape {} ", input_desc[i].shape);
  } 

  for (int i=0; i<output_desc.size(); i++){
    output_desc[i].shape[0] *= batch_per_step;
    output_desc[i].size_in_bytes *= batch_per_step;
  } 

  input_memory = examples::allocateHostInputData(input_desc);
  output_memory = examples::allocateHostInputData(output_desc);

  for (int i = 0; i < input_desc.size(); i++) {
    auto desc = input_desc[i];
    MMDEPLOY_INFO("input desc dtype {} ", desc.data_type);

    mmdeploy::DataType dtype = ipu_type_convert(desc.data_type);

    input_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        dtype,
        desc.shape,
        desc.name,
    });
  }

  for (int i = 0; i < output_desc.size(); i++) {
    auto desc = output_desc[i];
    MMDEPLOY_INFO("output desc dtype {} ", desc.data_type);
    mmdeploy::DataType dtype = ipu_type_convert(desc.data_type);

    output_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        dtype,
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
  MMDEPLOY_INFO("ipu device running forward ");
  // OUTCOME_TRY(stream_.Wait());

  {
    // copy input to itensor buffer

    for (auto& tensor : input_tensors_) {
      const auto& name = tensor.desc().name;
      copy_input(tensor, input_memory[name]);
    }
  }

  {
    struct timeval start, end;
    // for (int i = 0; i < 50; i++) {
    gettimeofday(&start, 0);
    model_runner->execute(examples::toInputMemoryView(input_memory),
                          examples::toOutputMemoryView(output_memory));
    gettimeofday(&end, 0);
    long sec = end.tv_sec - start.tv_sec;
    long micro = end.tv_usec - start.tv_usec;
    double elapsed = sec + micro * 1e-6;
    // output_desc = model_runner->getExecuteOutputs();
    // sleep(30);
    MMDEPLOY_INFO("ipu inference done, time elapsed {} sec", elapsed);
    // }
    // if (!success) {
    //   MMDEPLOY_ERROR("IPU Inference error: {}",
    //   std::string(zdl::DlSystem::getLastErrorString())); return Status(eFail);
    // }
  }

  {
    for (int i = 0; i < output_tensors_.size(); i++) {
      auto to_tensor = output_tensors_[i];
      auto name = to_tensor.desc().name;

      copy_output(output_memory[name], to_tensor);
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

// class IPUNetCreator : public Creator<Net> {
//  public:
//   const char* GetName() const override { return "ipu"; }
//   int GetVersion() const override { return 0; }
//   std::unique_ptr<Net> Create(const Value& args) override {
//     auto p = std::make_unique<IPUNet>();
//     if (auto r = p->Init(args)) {
//       return p;
//     } else {
//       MMDEPLOY_ERROR("error creating IPUNet: {}", r.error().message().c_str());
//       return nullptr;
//     }
//   }
// };

// REGISTER_MODULE(Net, IPUNetCreator);
static std::unique_ptr<Net> Create(const Value& args) {
  auto p = std::make_unique<IPUNet>();
  if (auto r = p->Init(args)) {
    return p;
  } else {
    MMDEPLOY_ERROR("error creating IPUnet: {}", r.error().message().c_str());
    return nullptr;
  }
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Net, (ipu, 0), Create);

}  // namespace mmdeploy
