// Copyright (c) OpenMMLab. All rights reserved.

#include "snpe_net.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

SNPENet::~SNPENet() {}

std::string SNPENet::ShapeStr(zdl::DlSystem::ITensor* pTensor) {
  std::string str;

  str += "[";
  auto shape = pTensor->getShape();
  for (int i = 0; i < shape.rank(); ++i) {
    str += std::to_string(shape[i]);
    str += ",";
  }
  str += ']';
  return str;
}

void SNPENet::Build(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                    zdl::DlSystem::Runtime_t runtime, zdl::DlSystem::RuntimeList runtimeList,
                    bool useUserSuppliedBuffers, zdl::DlSystem::PlatformConfig platformConfig) {
  zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

  if (runtimeList.empty()) {
    runtimeList.add(runtime);
  }

  snpe_ =
      snpeBuilder.setOutputLayers({})
          .setRuntimeProcessorOrder(runtimeList)
          .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
          .setPlatformConfig(platformConfig)
          .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE)
          .build();
  return;
}

void SNPENet::copy_output(const zdl::DlSystem::ITensor* from, Tensor& to) {
  auto hwc_to_chw = [](const zdl::DlSystem::TensorShape& shape) -> bool {
    if (shape.rank() != 4 || (shape[1] == 1 && shape[2] > 1 && shape[3] > 1)) {
      return false;
    }
    return true;
  };

  auto output_shape = from->getShape();

  if (to.size() != from->getSize()) {
    TensorShape tensor_shape;
    for (int j = 0; j < output_shape.rank(); ++j) {
      tensor_shape.push_back(output_shape[j]);
    }

    if (hwc_to_chw(output_shape)) {
      auto tmp = output_shape[3];
      output_shape[3] = output_shape[1];
      output_shape[1] = tmp;
    }
    to.Reshape(tensor_shape);
  }

  float* pto = to.data<float>();

  if (output_shape.rank() != 4 ||
      (output_shape[1] == 1 && output_shape[2] > 1 && output_shape[3] > 1)) {
    // skip [1,1,w>1,h>1] for segmentation task
    for (auto it = from->cbegin(); it != from->cend(); ++it, ++pto) {
      *pto = *it;
    }
  } else {
    const int channel = output_shape[1];
    const int panel = output_shape[2] * output_shape[3];

    int i = 0;
    // HWC to CHW
    for (auto it = from->cbegin(); it != from->cend(); ++it, ++i) {
      int channel_idx = i % channel;
      int panel_idx = i / channel;
      pto[channel_idx * panel + panel_idx] = *it;
    }
  }
  return;
}

void SNPENet::copy_input(const Tensor& from, zdl::DlSystem::ITensor* to) {
  if (from.size() != to->getSize()) {
    MMDEPLOY_ERROR("input tensor size not match");
    return;
  }

  const float* pfrom = from.data<float>();

  auto input_shape = to->getShape();
  if (input_shape.rank() == 4) {
    const int channel = input_shape[3];
    const int panel = input_shape[1] * input_shape[2];

    int i = 0;
    // CHW to HWC
    for (auto it = to->begin(); it != to->end(); ++it, ++i) {
      int channel_index = i % channel;
      int panel_index = (i / channel) % panel;

      *it = pfrom[channel_index * panel + panel_index];
    }

  } else {
    for (auto it = to->begin(); it != to->end(); ++it, ++pfrom) {
      *it = *pfrom;
    }
  }
}

Result<void> SNPENet::Init(const Value& args) {
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
  char* model_ptr = const_cast<char*>(content.data());
  container_ =
      zdl::DlContainer::IDlContainer::open(reinterpret_cast<uint8_t*>(model_ptr), content.size());
  if (container_ == nullptr) {
    MMDEPLOY_ERROR("Load .dlc failed: {}", config.net);
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

Result<void> SNPENet::Deinit() { return success(); }

Result<void> SNPENet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

Result<Span<Tensor>> SNPENet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> SNPENet::GetOutputTensors() { return output_tensors_; }

Result<void> SNPENet::Forward() {
  OUTCOME_TRY(stream_.Wait());

  {
    // copy input to itensor buffer
    for (auto& tensor : input_tensors_) {
      const auto& name = tensor.desc().name;
      auto pbuffer = input_tensor_map_.getTensor(name.c_str());

      copy_input(tensor, pbuffer);
    }
  }

  // A tensor map for SNPE execution outputs
  zdl::DlSystem::TensorMap output_map;
  {
    // real inference
    bool success = snpe_->execute(input_tensor_map_, output_map);
    if (!success) {
      MMDEPLOY_ERROR("snpe Inference error: {}", std::string(zdl::DlSystem::getLastErrorString()));
      return Status(eFail);
    }
  }

  {
    // extract output buffer to tensor
    auto names = output_map.getTensorNames();
    for (size_t i = 0; i < names.size(); ++i) {
      const zdl::DlSystem::ITensor* pbuffer = output_map.getTensor(names.at(i));

      auto& tensor = output_tensors_[i];
      copy_output(pbuffer, tensor);
    }
  }
  return success();
}

class SNPENetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "snpe"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    auto p = std::make_unique<SNPENet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating SNPENet: {}", r.error().message().c_str());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, SNPENetCreator);

}  // namespace mmdeploy
