// Copyright (c) OpenMMLab. All rights reserved.

#include "snpe_net.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

SNPENet::~SNPENet() {
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

  container_ = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(config.net));
  if (container_ == nullptr) {
    MMDEPLOY_ERROR("Load .dlc failed: {}", config.net);
    return Status(eInvalidArgument);
  }

  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
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
    input_tensor_map_.add(inputTensorNames.at(i), inputs_internal_[i].get());

    input_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        DataType::kFLOAT,
        {},
        std::string(inputTensorNames.at(i)),
    });
  }

  const auto& outputTensorNamesRef = snpe_ ->getOutputTensorNames();
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

Result<void> SNPENet::Deinit() {
  return success();
}

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

      float* from = tensor.data<float>();
      std::vector<float> vec = {from, from + tensor.size()};
      std::copy(vec.begin(), vec.end(), pbuffer->begin());
    }
  }

  // A tensor map for SNPE execution outputs
  zdl::DlSystem::TensorMap output_map;
  {
    // real inference
    bool success = snpe_->execute(input_tensor_map_, output_map);
    if (!success) {
      MMDEPLOY_ERROR("snpe Inference error: {}", std::string(zdl::DlSystem::getLastErrorString()));
    }
  }

  {
    // extract output buffer to tensor
    auto names = output_map.getTensorNames();
    for (size_t i = 0; i < names.size(); ++i) {
      zdl::DlSystem::ITensor* pbuffer = output_map.getTensor(names.at(i));

      auto& tensor = output_tensors_[i];
      auto out_shape = pbuffer->getShape();

      TensorShape tensor_shape;
      for (int j = 0; j < out_shape.rank(); ++j) {
        tensor_shape.push_back(out_shape[i]);
      }
      tensor.Reshape(tensor_shape);

      float* to = tensor.data<float>();

      for (auto it = pbuffer->cbegin(); it != pbuffer->cend(); ++it) {
        *to = *it;
        ++to;
      }
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
