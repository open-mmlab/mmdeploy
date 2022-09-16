// Copyright (c) OpenMMLab. All rights reserved.
#include "rknn_net.h"

#include <stdio.h>

#include <fstream>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

template <typename T>
Result<std::unique_ptr<T>> rknn_try(T* v) {
  if (v) {
    return success(v);
  }
  return Status(eFail);
}

static unsigned char* load_model(const char* filename, int* model_size) {
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int model_len = ftell(fp);
  unsigned char* model = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp) {
    fclose(fp);
  }
  return model;
}

static Result<DataType> ConvertElementType(InferenceEngine::Precision prec) {}

static Result<InferenceEngine::Precision::ePrecision> ConvertPrecision(DataType type) {}

static Result<std::string> ConvertDeviceName(const Device& device) {
  if (device.is_host()) {
    return "CPU";
  }
  return Status(eNotSupported);
}

Result<void> RKNNNet::Init(const Value& args) {
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();
  if (!device_.is_host()) {
    return Status(eNotSupported);
  }

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));

  std::string content;
  OUTCOME_TRY(content, model.ReadFile(config.net));
  unsigned char* model_ptr = const_cast<unsigned char*>(content.data());
  int ret = rknn_init(&ctx_, model_ptr, content.size(), 0, NULL);
  if (ret < 0) {
    MMDEPLOY_ERROR("Load .rknn failed: {}", config.net);
    return Status(eInvalidArgument);
  }

  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::GPU;
  if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
    MMDEPLOY_WARN("Selected runtime not present. Falling back to CPU.\n");
    runtime = zdl::DlSystem::Runtime_t::CPU;
  }

  zdl::DlSystem::RuntimeList runtimeList;
  // Add CPU backend to support fallback
  runtimeList.add(zdl::DlSystem::Runtime_t::CPU);
  runtimeList.add(runtime);
  zdl::DlSystem::PlatformConfig platformConfig;
  Build(container_, runtime, runtimeList, false, platformConfig);

  // init internal input tensor list
  const auto& inputTensorNamesRef = snpe_->getInputTensorNames();
  const auto& inputTensorNames = *inputTensorNamesRef;
  inputs_internal_.resize(inputTensorNames.size());

  for (int i = 0; i < inputTensorNames.size(); ++i) {
    const auto& inputShape_opt = snpe_->getInputDimensions(inputTensorNames.at(i));
    const auto& inputShape = *inputShape_opt;

    inputs_internal_[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

    std::string info =
        std::string(inputTensorNames.at(i)) + " shape: " + ShapeStr(inputs_internal_[i].get());
    MMDEPLOY_INFO(info);

    input_tensor_map_.add(inputTensorNames.at(i), inputs_internal_[i].get());

    input_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        DataType::kFLOAT,
        {},
        std::string(inputTensorNames.at(i)),
    });
  }

  const auto& outputTensorNamesRef = snpe_->getOutputTensorNames();
  const auto& outputTensorNames = *outputTensorNamesRef;
  for (int i = 0; i < outputTensorNames.size(); ++i) {
    output_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        DataType::kFLOAT,
        {},
        std::string(outputTensorNames.at(i)),
    });
  }

  return success();
}

Result<void> RKNNNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

Result<void> RKNNNet::Deinit() { return success(); }

Result<Span<Tensor>> RKNNNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> RKNNNet::GetOutputTensors() { return output_tensors_; }

Result<void> RKNNNet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

Result<void> RKNNNet::Forward() {
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

class RKNNNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "rknn"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    try {
      auto p = std::make_unique<RKNNNet>();
      if (auto r = p->Init(args)) {
        return p;
      } else {
        MMDEPLOY_ERROR("error creating RKNNNet: {}", r.error().message().c_str());
        return nullptr;
      }
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("unhandled exception when creating RKNNNet: {}", e.what());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, RKNNNetCreator);

}  // namespace mmdeploy
