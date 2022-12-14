// Copyright (c) OpenMMLab. All rights reserved.
#include "openvino_net.h"

#include <stdio.h>

#include <fstream>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy::framework {

template <typename T>
Result<std::unique_ptr<T>> openvino_try(T* v) {
  if (v) {
    return success(v);
  }
  return Status(eFail);
}

static Result<DataType> ConvertElementType(InferenceEngine::Precision prec) {
  auto type = InferenceEngine::Precision::ePrecision(prec);
  switch (type) {
    case InferenceEngine::Precision::ePrecision::FP32:
      return DataType::kFLOAT;
    case InferenceEngine::Precision::ePrecision::FP16:
      return DataType::kHALF;
    case InferenceEngine::Precision::ePrecision::I8:
      return DataType::kINT8;
    case InferenceEngine::Precision::ePrecision::I32:
      return DataType::kINT32;
    case InferenceEngine::Precision::ePrecision::I64:
      return DataType::kINT64;
    default:
      MMDEPLOY_ERROR("unsupported InferenceEngine Precision: {}", static_cast<int>(type));
      return Status(eNotSupported);
  }
}

static Result<InferenceEngine::Precision::ePrecision> ConvertPrecision(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return InferenceEngine::Precision::ePrecision::FP32;
    case DataType::kHALF:
      return InferenceEngine::Precision::ePrecision::FP16;
    case DataType::kINT8:
      return InferenceEngine::Precision::ePrecision::I8;
    case DataType::kINT32:
      return InferenceEngine::Precision::ePrecision::I32;
    case DataType::kINT64:
      return InferenceEngine::Precision::ePrecision::I64;
    default:
      MMDEPLOY_ERROR("unsupported DataType: {}", static_cast<int>(type));
      return Status(eNotSupported);
  }
}

static Result<std::string> ConvertDeviceName(const Device& device) {
  if (device.is_host()) {
    return "CPU";
  }
  return Status(eNotSupported);
}

Result<void> OpenVINONet::Init(const Value& args) {
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();

  if (!device_.is_host()) {
    return Status(eNotSupported);
  }

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));

  // TODO: read network with stream
  // save xml and bin to temp file
  auto tmp_dir = fs::temp_directory_path();
  std::string tmp_xml = (tmp_dir / fs::path("tmp.xml")).string();
  std::string tmp_bin = (tmp_dir / fs::path("tmp.bin")).string();
  OUTCOME_TRY(auto raw_xml, model.ReadFile(config.net));
  OUTCOME_TRY(auto raw_bin, model.ReadFile(config.weights));

  try {
    std::ofstream xml_out(tmp_xml, std::ios::binary);
    xml_out << raw_xml;
    xml_out.close();
    std::ofstream bin_out(tmp_bin, std::ios::binary);
    bin_out << raw_bin;
    bin_out.close();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating tmp xml/bin: {}", e.what());
    return Status(eFail);
  }

  try {
    // create cnnnetwork
    core_ = InferenceEngine::Core();
    network_ = core_.ReadNetwork(tmp_xml, tmp_bin);

    // set input tensor
    InferenceEngine::InputsDataMap input_info = network_.getInputsInfo();
    for (auto& item : input_info) {
      auto input_data = item.second;
      const auto& input_name = input_data->name();
      OUTCOME_TRY(auto data_type, ConvertElementType(input_data->getPrecision()));
      const auto& size_vector = input_data->getTensorDesc().getDims();
      TensorShape shape{size_vector.begin(), size_vector.end()};
      input_tensors_.emplace_back(TensorDesc{device_, data_type, shape, input_name});
    }

    // set output tensor
    InferenceEngine::OutputsDataMap output_info = network_.getOutputsInfo();
    for (auto& item : output_info) {
      auto output_data = item.second;
      const auto& output_name = output_data->getName();
      OUTCOME_TRY(auto data_type, ConvertElementType(output_data->getPrecision()));
      const auto& size_vector = output_data->getDims();
      TensorShape shape{size_vector.begin(), size_vector.end()};
      output_tensors_.emplace_back(TensorDesc{device_, data_type, shape, output_name});
    }

    // create request
    net_config_ =
        std::map<std::string, std::string>{{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
                                            InferenceEngine::PluginConfigParams::YES}};
    OUTCOME_TRY(auto device_str, ConvertDeviceName(device_));
    auto executable_network = core_.LoadNetwork(network_, device_str, net_config_);
    request_ = executable_network.CreateInferRequest();

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating OpenVINO: {}", e.what());
    return Status(eFail);
  }
  return success();
}

Result<void> OpenVINONet::ForwardAsync(Event* event) { return Status(eNotSupported); }

Result<void> OpenVINONet::Deinit() { return success(); }

Result<Span<Tensor>> OpenVINONet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> OpenVINONet::GetOutputTensors() { return output_tensors_; }

Result<void> OpenVINONet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

static Result<void> SetBlob(InferenceEngine::InferRequest& request, Tensor& tensor) {
  const auto& input_name = tensor.desc().name;

  const auto& desc = tensor.desc();
  const auto& shape = desc.shape;
  InferenceEngine::SizeVector size_vector{shape.begin(), shape.end()};
  OUTCOME_TRY(auto prec, ConvertPrecision(desc.data_type));
  InferenceEngine::TensorDesc ie_desc(prec, size_vector, InferenceEngine::Layout::NCHW);

  // TODO: find a better way instead of switch case
  switch (desc.data_type) {
    case DataType::kFLOAT:
      request.SetBlob(input_name,
                      InferenceEngine::make_shared_blob<float>(ie_desc, tensor.data<float>()));
      break;
    case DataType::kINT8:
      request.SetBlob(input_name,
                      InferenceEngine::make_shared_blob<int8_t>(ie_desc, tensor.data<int8_t>()));
      break;
    case DataType::kINT32:
      request.SetBlob(input_name,
                      InferenceEngine::make_shared_blob<int32_t>(ie_desc, tensor.data<int32_t>()));
      break;
    case DataType::kINT64:
      request.SetBlob(input_name,
                      InferenceEngine::make_shared_blob<int64_t>(ie_desc, tensor.data<int64_t>()));
      break;
    default:
      MMDEPLOY_ERROR("unsupported DataType: {}", static_cast<int>(desc.data_type));
      return Status(eNotSupported);
  }
  return success();
}

static Result<void> GetBlob(InferenceEngine::InferRequest& request, Tensor& tensor,
                            Stream& stream) {
  const auto& desc = tensor.desc();
  const auto& output_name = desc.name;
  const auto device = desc.device;
  const auto data_type = desc.data_type;
  const auto& output = request.GetBlob(output_name);
  const auto& size_vector = output->getTensorDesc().getDims();
  TensorShape shape{size_vector.begin(), size_vector.end()};

  InferenceEngine::MemoryBlob::CPtr moutput =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
  auto moutputHolder = moutput->rmap();
  std::shared_ptr<void> data(const_cast<void*>(moutputHolder.as<const void*>()), [](void*) {});

  Tensor blob_tensor = {TensorDesc{device, data_type, shape, output_name}, data};
  if (!std::equal(blob_tensor.shape().begin(), blob_tensor.shape().end(), tensor.shape().begin()))
    tensor.Reshape(shape);
  OUTCOME_TRY(tensor.CopyFrom(blob_tensor, stream));

  return success();
}

Result<void> OpenVINONet::Forward() {
  OUTCOME_TRY(stream_.Wait());

  // reshape network if shape does not match
  bool need_reshape = false;
  auto input_shapes = network_.getInputShapes();
  for (auto& tensor : input_tensors_) {
    const auto& input_name = tensor.desc().name;
    const auto& tensor_shape = tensor.desc().shape;
    auto& size_vector = input_shapes[input_name];
    bool shape_changed = !std::equal(size_vector.begin(), size_vector.end(), tensor_shape.begin(),
                                     [](size_t a, int64_t b) { return a == size_t(b); });
    need_reshape |= shape_changed;
    if (shape_changed)
      size_vector = InferenceEngine::SizeVector{tensor_shape.begin(), tensor_shape.end()};
  }

  if (need_reshape) {
    network_.reshape(input_shapes);
    OUTCOME_TRY(auto device_str, ConvertDeviceName(device_));
    auto executable_network = core_.LoadNetwork(network_, device_str, net_config_);
    request_ = executable_network.CreateInferRequest();
  }

  // fill input into request
  for (auto& tensor : input_tensors_) {
    OUTCOME_TRY(SetBlob(request_, tensor));
  }

  request_.StartAsync();
  request_.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);

  // read output from request
  for (auto& tensor : output_tensors_) {
    OUTCOME_TRY(GetBlob(request_, tensor, stream_));
  }
  OUTCOME_TRY(stream_.Wait());

  return success();
}

static std::unique_ptr<Net> Create(const Value& args) {
  try {
    auto p = std::make_unique<OpenVINONet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating OpenVINONet: {}", r.error().message().c_str());
      return nullptr;
    }
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating OpenVINONet: {}", e.what());
    return nullptr;
  }
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Net, (openvino, 0), Create);

}  // namespace mmdeploy::framework
