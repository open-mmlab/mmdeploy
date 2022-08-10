// Copyright (c) OpenMMLab. All rights reserved.

#include "torch_net.h"

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

#include "torchvision/vision.h"

namespace mmdeploy {

TorchNet::~TorchNet() = default;

class InferenceMode {
  at::AutoNonVariableTypeMode guard_;
};

static Result<torch::ScalarType> FromDataType(DataType data_type) {
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

static Result<DataType> ToDataType(torch::ScalarType scalar_type) {
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
    module_.to(*torch_device_);
    auto forward = module_.get_method("forward");
    auto arguments = forward.function().getSchema().arguments();
    for (const auto& arg : arguments) {
      MMDEPLOY_INFO("arg: {}", arg.name());
    }
    auto returns = forward.function().getSchema().returns();
    for (const auto& ret : returns) {
      MMDEPLOY_INFO("ret: {}", ret.name());
    }

    auto ToDesc = [&](torch::jit::Value* v) {
      MMDEPLOY_ERROR("{}", v->debugNameBase());
      // torch::jit::Tensor
      // torch::jit::NamedValue
      // MMDEPLOY_ERROR("{}", v->node()->cast<torch::jit::TensorType>());
      return TensorDesc{device_, DataType::kFLOAT, {}, v->debugNameBase()};
    };

    auto inputs = forward.graph()->inputs();

    for (int i = 1; i < inputs.size(); ++i) {
      MMDEPLOY_ERROR("{}", inputs[i]->type()->annotation_str());
      if (inputs[i]->type()->kind() == c10::TypeKind::TensorType) {
        input_tensor_.emplace_back(ToDesc(inputs[i]));
      } else {
        MMDEPLOY_ERROR("Unsupported input type: {}", typeKindToString(inputs[i]->type()->kind()));
        return Status(eNotSupported);
      }
    }

    auto outputs = forward.graph()->outputs();
    for (const auto& output : outputs) {
      // MMDEPLOY_ERROR("{}", typeKindToString(output->type()->kind()));
      MMDEPLOY_ERROR("{}", output->type()->annotation_str());
      auto kind = output->type()->kind();
      if (kind == c10::TypeKind::TensorType) {
        output_tensor_.emplace_back(ToDesc(output));
      } else if (output->type()->kind() == c10::TypeKind::TupleType) {
        for (const auto& v : output->node()->inputs()) {
          if (v->type()->kind() == c10::TypeKind::TensorType) {
            output_tensor_.emplace_back(ToDesc(v));
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
  OUTCOME_TRY(stream_.Wait());
  try {
    InferenceMode guard;
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
    } else if (outputs.isTensorList()) {
      auto tensor_vector = outputs.toTensorVector();
      for (size_t i = 0; i < tensor_vector.size(); ++i) {
        OUTCOME_TRY(output_tensor_[i],
                    FromTorchTensor(outputs.toTensor(), output_tensor_[i].name()));
      }
    } else if (outputs.isTuple()) {
      auto tuple = outputs.toTuple();
      size_t index = 0;
      for (const auto& x : tuple->elements()) {
        MMDEPLOY_ERROR(toString(x.toTensor().device()));
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

}  // namespace mmdeploy
