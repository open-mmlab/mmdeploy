// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/net/torchscript/torch_net.h"

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "torch/torch.h"

#if MMDEPLOY_USE_CUDA
#include "c10/cuda/CUDAGuard.h"
#include "c10/cuda/CUDAStream.h"
#endif

#if MMDEPLOY_USE_TORCHVISION
#include "torchvision/vision.h"
MMDEPLOY_API void _mmdeploy_force_link_torchvision() { vision::detail::_register_ops(); }
#endif

namespace mmdeploy::framework {

namespace {

class InferenceMode {
#if TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 10
  c10::InferenceMode guard_;
#else
  at::AutoNonVariableTypeMode guard_;
#endif
};

class StreamGuard {
 public:
  StreamGuard(const torch::Device& device, Stream stream)
      : device_(device), stream_(std::move(stream)), device_guard_(device) {
    stream_.Wait().value();
  }

  ~StreamGuard() {
#if MMDEPLOY_USE_CUDA
    auto device = stream_.GetDevice();
    if (device.is_device()) {
      Stream stream(device, (cudaStream_t)c10::cuda::getCurrentCUDAStream(device_.index()));
      stream.Wait().value();
    }
#endif
  }

 private:
  torch::Device device_;
  Stream stream_;
  c10::DeviceGuard device_guard_;
};

Result<torch::ScalarType> FromDataType(DataType data_type) {
  switch (data_type) {
    case DataType::kFLOAT:
      return torch::ScalarType::Float;
    case DataType::kHALF:
      return torch::ScalarType::Half;
    case DataType::kINT32:
      return torch::ScalarType::Int;
    case DataType::kINT64:
      return torch::ScalarType::Long;
    case DataType::kINT8:
      return torch::ScalarType::Char;
    default:
      MMDEPLOY_ERROR("Unsupported mmdeploy::DataType: {}", to_string(data_type));
      return Status(eNotSupported);
  }
}

Result<DataType> ToDataType(torch::ScalarType scalar_type) {
  switch (scalar_type) {
    case torch::ScalarType::Float:
      return DataType::kFLOAT;
    case torch::ScalarType::Half:
      return DataType::kHALF;
    case torch::ScalarType::Int:
      return DataType::kINT32;
    case torch::ScalarType::Long:
      return DataType::kINT64;
    case torch::ScalarType::Char:
      return DataType::kINT8;
    default:
      MMDEPLOY_ERROR("Unsupported torch::ScalarType: {}", toString(scalar_type));
      return Status(eNotSupported);
  }
}

}  // namespace

TorchNet::~TorchNet() = default;

Result<void> TorchNet::Init(const Value& cfg) {
  auto& context = cfg["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();

  auto name = cfg["name"].get<std::string>();
  auto model = context["model"].get<Model>();

  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  OUTCOME_TRY(auto bytes, model.ReadFile(config.net));

  auto platform = Platform(device_.platform_id());
  auto device_name = platform.GetPlatformName();

  try {
    {
      using namespace std::string_literals;
      if (device_name == "cpu"s) {
        torch_device_ = torch::Device(device_name);
      } else {
        torch_device_ = torch::Device(device_name + ":"s + std::to_string(device_.device_id()));
      }
    }
    std::istringstream iss(bytes);
    InferenceMode guard;
    module_ = torch::jit::load(iss);
    module_.eval();
    module_.to(*torch_device_);
    auto forward = module_.get_method("forward");

    auto ToDesc = [&](torch::jit::Value* value, const char* type, int index) {
      MMDEPLOY_INFO("Found {}: {}", type, value->debugNameBase());
      return TensorDesc{device_, DataType::kFLOAT, {}, "#" + std::to_string(index)};
    };

    auto inputs = forward.graph()->inputs();
    int input_count = 0;
    for (int i = 1; i < inputs.size(); ++i) {
      if (inputs[i]->type()->kind() == c10::TypeKind::TensorType) {
        input_tensor_.emplace_back(ToDesc(inputs[i], "input", input_count++));
      } else {
        MMDEPLOY_ERROR("Unsupported input type: {}", typeKindToString(inputs[i]->type()->kind()));
        return Status(eNotSupported);
      }
    }

    auto outputs = forward.graph()->outputs();
    int output_count = 0;
    for (const auto& output : outputs) {
      auto kind = output->type()->kind();
      if (kind == c10::TypeKind::TensorType) {
        output_tensor_.emplace_back(ToDesc(output, "output", output_count++));
      } else if (output->type()->kind() == c10::TypeKind::TupleType) {
        for (const auto& v : output->node()->inputs()) {
          if (v->type()->kind() == c10::TypeKind::TensorType) {
            output_tensor_.emplace_back(ToDesc(v, "output", output_count++));
          } else {
            MMDEPLOY_ERROR("Unsupported output type: {}", typeKindToString(v->type()->kind()));
            return Status(eNotSupported);
          }
        }
      } else {
        MMDEPLOY_ERROR("Unsupported output type: {}", typeKindToString(kind));
      }
    }
    return success();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
    return Status(eFail);
  }
}

Result<void> TorchNet::Deinit() { return success(); }
Result<Span<Tensor>> TorchNet::GetInputTensors() { return input_tensor_; }
Result<Span<Tensor>> TorchNet::GetOutputTensors() { return output_tensor_; }

Result<void> TorchNet::Reshape(Span<TensorShape> input_shapes) {
  if (input_shapes.size() != input_tensor_.size()) {
    return Status(eInvalidArgument);
  }
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensor_[i].Reshape(input_shapes[i]);
  }
  return success();
}

Result<void> TorchNet::Forward() {
  try {
    StreamGuard stream_guard(*torch_device_, stream_);
    InferenceMode inference_guard;
    std::vector<torch::jit::IValue> inputs;
    for (auto& v : input_tensor_) {
      OUTCOME_TRY(auto data_type, FromDataType(v.data_type()));
      auto tensor = torch::from_blob(v.data(), v.shape(),
                                     c10::TensorOptions(*torch_device_).dtype(data_type));
      inputs.emplace_back(tensor);
    }
    auto outputs = module_.forward(inputs);
    if (outputs.isTensor()) {
      OUTCOME_TRY(output_tensor_[0], FromTorchTensor(outputs.toTensor(), output_tensor_[0].name()));
    } else if (outputs.isTuple()) {
      auto tuple = outputs.toTuple();
      size_t index = 0;
      for (const auto& x : tuple->elements()) {
        OUTCOME_TRY(output_tensor_[index],
                    FromTorchTensor(x.toTensor(), output_tensor_[index].name()));
        ++index;
      }
    } else {
      MMDEPLOY_ERROR("{}", toString(outputs.type()));
      return Status(eNotSupported);
    }
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
    return Status(eFail);
  }
  return success();
}
Result<void> TorchNet::ForwardAsync(Event* event) { return success(); }

Result<Tensor> TorchNet::FromTorchTensor(const torch::Tensor& tensor, const std::string& name) {
  OUTCOME_TRY(auto data_type, ToDataType(tensor.scalar_type()));
  auto shape = tensor.sizes();
  TensorDesc desc{device_, data_type, {shape.begin(), shape.end()}, name};
  return Tensor(desc, std::shared_ptr<void>(tensor.data_ptr(), [tensor](auto) {}));
}

class TorchNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "torchscript"; }
  std::unique_ptr<Net> Create(const Value& cfg) override {
    auto p = std::make_unique<TorchNet>();
    if (auto status = p->Init(cfg)) {
      return p;
    } else {
      MMDEPLOY_ERROR("Failed to created TorchNet with config: {}", cfg);
    }
    return nullptr;
  }
};

REGISTER_MODULE(Net, TorchNetCreator);

}  // namespace mmdeploy::framework
