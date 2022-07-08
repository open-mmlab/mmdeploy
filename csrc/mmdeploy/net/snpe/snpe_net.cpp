// Copyright (c) OpenMMLab. All rights reserved.

#include "ncnn_net.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

SNPENet::~SNPENet() {}

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
  }

  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::GPU;
  if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
    MMDEPLOY_WARN("Selected runtime not present. Falling back to CPU.\n");
    runtime = zdl::DlSystem::Runtime_t::CPU;
  }

  zdl::DlSystem::RuntimeList runtimeList;
  runtimeList.add(runtime);
  zdl::DlSystem::PlatformConfig platformConfig;
  snpe_ = SetBuilderOptions(container_, runtime, runtimeList, false,
                           platformConfig, false);

  // init internal input tensor list
  const auto& inputTensorNamesRef = snpe->getInputTensorNames();
  const auto& inputTensorNames = *inputTensorNamesRef;
  inputs_internal_.resize(inputTensorNames.size());

  for (int i = 0; i < inputTensorNames.size(); ++i) {
    const auto& inputShape_opt = snpe->getInputDimensions(inputTensorNames.at(i));
    const auto& inputShape = *inputShape_opt;

    inputs_internal_[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    input_tensor_map_.add(tensor.name().c_str(), inputTensors[i].get());
  }

  return success();
}

Result<void> SNPENet::Deinit() {
  input_tensor_map_.clear();
  container_.reset();
  snpe_.reset();
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

  const int LEN = inputs_internal_.size();
  for (int i = 0; i < LEN; ++i) {
    float *from = input_tensors_[i].data<float>();
    std::vector<float> vec = {from, from + inpute_tensors_[i].size()};
    std::copy(vec.begin(), vec.end(), input_tensors[i]->begin());
  }

  bool success = snpe->execute(inputTensorMap, outputTensorMap);
  if (! success) {
    MMDEPLOY_ERROR("snpe Inference error: {}", std::string(zdl::DlSystem::getLastErrorString()));
  }

  // extract result
  auto out_names = outputTensorMap.getTensorNames();
  for (size_t i = 0; i < out_names.size(); ++i) {
    const char* name = out_names.at(i);
    zdl::DlSystem::ITensor* pTensor = outputTensorMap.getTensor(name);

    size_t data_size = sizeof(float) * pTensor->getSize();

    auto& tensor = output_tensors_[i];
    auto& shape = pTensor->getShape();
    switch (shape.rank())
    {
    case 1:
      tensor.Reshape({shape[0]}):
      break;
    case 2:
      tensor.Reshape({shape[0], shape[1]}):
      break;
    case 3:
      tensor.Reshape({shape[0], shape[1], shape[2]}):
      break;
    case 4:
      tensor.Reshape({shape[0], shape[1], shape[2], shape[3]}):
    default:
      break;
    }

    float* to = tensor.data<float>();
    int j = 0;
    for (auto it = pTensor->cbegin(); it != pTensor->cend(); ++it, ++j) {
      to[j] = *it;
    }
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
