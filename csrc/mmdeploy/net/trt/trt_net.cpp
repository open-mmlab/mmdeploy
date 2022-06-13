// Copyright (c) OpenMMLab. All rights reserved.

#include "trt_net.h"

#include <sstream>

#include "core/logger.h"
#include "core/model.h"
#include "core/module.h"
#include "core/utils/formatter.h"

namespace mmdeploy {

namespace trt_detail {

class TRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINFO:
        // MMDEPLOY_INFO("TRTNet: {}", msg);
        break;
      case Severity::kWARNING:
        MMDEPLOY_WARN("TRTNet: {}", msg);
        break;
      case Severity::kERROR:
      case Severity::kINTERNAL_ERROR:
        MMDEPLOY_ERROR("TRTNet: {}", msg);
        break;
      default:
        break;
    }
  }
  static TRTLogger& get() {
    static TRTLogger trt_logger{};
    return trt_logger;
  }
};

nvinfer1::Dims to_dims(const TensorShape& shape) {
  nvinfer1::Dims dims{};
  dims.nbDims = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    dims.d[i] = shape[i];
  }
  return dims;
}

TensorShape to_shape(const nvinfer1::Dims& dims) {
  TensorShape shape(dims.nbDims);
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = dims.d[i];
  }
  return shape;
}

}  // namespace trt_detail

std::string to_string(const nvinfer1::Dims& dims) {
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i) ss << ", ";
    ss << dims.d[i];
  }
  ss << ")";
  return ss.str();
}

static inline Result<void> trt_try(bool code, const char* msg = nullptr, Status e = Status(eFail)) {
  if (code) {
    return success();
  }
  if (msg) {
    MMDEPLOY_ERROR("{}", msg);
  }
  return e;
}

#define TRT_TRY(...) OUTCOME_TRY(trt_try(__VA_ARGS__))

TRTNet::~TRTNet() = default;

static Result<DataType> MapDataType(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return DataType::kFLOAT;
    case nvinfer1::DataType::kHALF:
      return DataType::kHALF;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
      return DataType::kINT8;
    case nvinfer1::DataType::kINT32:
      return DataType::kINT32;
    default:
      return Status(eNotSupported);
  }
}

Result<void> TRTNet::Init(const Value& args) {
  using namespace trt_detail;

  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  if (device_.is_host()) {
    MMDEPLOY_ERROR("TRTNet: device must be a GPU!");
    return Status(eNotSupported);
  }
  stream_ = context["stream"].get<Stream>();

  event_ = Event(device_);

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));

  OUTCOME_TRY(auto plan, model.ReadFile(config.net));

  TRTWrapper runtime = nvinfer1::createInferRuntime(TRTLogger::get());
  TRT_TRY(!!runtime, "failed to create TRT infer runtime");

  engine_ = runtime->deserializeCudaEngine(plan.data(), plan.size());
  TRT_TRY(!!engine_, "failed to deserialize TRT CUDA engine");

  TRT_TRY(engine_->getNbOptimizationProfiles() == 1, "only 1 optimization profile supported",
          Status(eNotSupported));

  auto n_bindings = engine_->getNbBindings();
  for (int i = 0; i < n_bindings; ++i) {
    auto binding_name = engine_->getBindingName(i);
    auto dims = engine_->getBindingDimensions(i);
    if (engine_->isShapeBinding(i)) {
      MMDEPLOY_ERROR("shape binding is not supported.");
      return Status(eNotSupported);
    }
    OUTCOME_TRY(auto dtype, MapDataType(engine_->getBindingDataType(i)));
    TensorDesc desc{device_, dtype, to_shape(dims), binding_name};
    if (engine_->bindingIsInput(i)) {
      MMDEPLOY_DEBUG("input binding {} {} {}", i, binding_name, to_string(dims));
      input_ids_.push_back(i);
      input_names_.emplace_back(binding_name);
      input_tensors_.emplace_back(desc, Buffer());
    } else {
      MMDEPLOY_DEBUG("output binding {} {} {}", i, binding_name, to_string(dims));
      output_ids_.push_back(i);
      output_names_.emplace_back(binding_name);
      output_tensors_.emplace_back(desc, Buffer());
    }
  }
  context_ = engine_->createExecutionContext();
  TRT_TRY(!!context_, "failed to create TRT execution context");

  context_->setOptimizationProfileAsync(0, static_cast<cudaStream_t>(stream_.GetNative()));
  OUTCOME_TRY(stream_.Wait());

  return success();
}

Result<void> TRTNet::Deinit() {
  context_.reset();
  engine_.reset();
  return success();
}

Result<void> TRTNet::Reshape(Span<TensorShape> input_shapes) {
  using namespace trt_detail;
  if (input_shapes.size() != input_tensors_.size()) {
    return Status(eInvalidArgument);
  }
  for (int i = 0; i < input_tensors_.size(); ++i) {
    auto dims = to_dims(input_shapes[i]);
    //    MMDEPLOY_ERROR("input shape: {}", to_string(dims));
    TRT_TRY(context_->setBindingDimensions(input_ids_[i], dims));
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  if (!context_->allInputDimensionsSpecified()) {
    MMDEPLOY_ERROR("not all input dimensions specified");
    return Status(eFail);
  }
  for (int i = 0; i < output_tensors_.size(); ++i) {
    auto dims = context_->getBindingDimensions(output_ids_[i]);
    //    MMDEPLOY_ERROR("output shape: {}", to_string(dims));
    output_tensors_[i].Reshape(to_shape(dims));
  }
  return success();
}

Result<Span<Tensor>> TRTNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> TRTNet::GetOutputTensors() { return output_tensors_; }

Result<void> TRTNet::Forward() {
  using namespace trt_detail;
  std::vector<void*> bindings(engine_->getNbBindings());

  for (int i = 0; i < input_tensors_.size(); ++i) {
    bindings[input_ids_[i]] = input_tensors_[i].data();
  }
  for (int i = 0; i < output_tensors_.size(); ++i) {
    bindings[output_ids_[i]] = output_tensors_[i].data();
  }

  auto event = GetNative<cudaEvent_t>(event_);
  auto status = context_->enqueueV2(bindings.data(), GetNative<cudaStream_t>(stream_), &event);
  TRT_TRY(status, "TRT forward failed", Status(eFail));
  OUTCOME_TRY(event_.Wait());

  return success();
}

Result<void> TRTNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

class TRTNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "tensorrt"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    auto p = std::make_unique<TRTNet>();
    if (p->Init(args)) {
      return p;
    }
    return nullptr;
  }
};

REGISTER_MODULE(Net, TRTNetCreator);

}  // namespace mmdeploy
