// Copyright (c) OpenMMLab. All rights reserved.
#include "ppl_net.h"

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#if PPL_NN_HAS_X86
#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/x86/engine_options.h"
#include "ppl/nn/engines/x86/ops.h"
#endif
#if PPL_NN_HAS_CUDA
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/engines/cuda/engine_options.h"
#include "ppl/nn/engines/cuda/ops.h"
#define PPL_CUDA_IMPORT_FROM_BUFFER 1
#endif
#if PPL_NN_HAS_RISCV
#include "ppl/nn/engines/riscv/engine_factory.h"
#include "ppl/nn/engines/riscv/engine_options.h"
#include "ppl/nn/engines/riscv/ops.h"
#endif

namespace mmdeploy {

Result<void> ppl_try(int code) {
  if (code == 0) {
    return success();
  }
  MMDEPLOY_ERROR("ppl error: {}", ppl::common::GetRetCodeStr(code));
  return Status(eFail);
}

template <typename T>
Result<std::unique_ptr<T>> ppl_try(T* v) {
  if (v) {
    return success(v);
  }
  return Status(eFail);
}

Tensor PPLNet::CreateInternalTensor(ppl::nn::Tensor* src, Device device) {
  const auto& desc = *src->GetShape();
  auto name = src->GetName();
  std::vector<int64_t> shape{desc.GetDims(), desc.GetDims() + desc.GetDimCount()};
  if (std::any_of(begin(shape), end(shape), [](auto x) { return x <= 0; })) {
    shape = {};
  }
  return TensorDesc{.device = device, .data_type = DataType::kFLOAT, .shape = shape, .name = name};
}

Result<void> PPLNet::Init(const Value& args) {
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();
  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();

  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  OUTCOME_TRY(auto onnx, model.ReadFile(config.net));

#if PPL_NN_HAS_CUDA
  if (device_.is_device()) {
    ppl::nn::cuda::RegisterBuiltinOpImpls();
    ppl::nn::cuda::EngineOptions options{};
    options.device_id = device_.device_id();
    options.mm_policy = ppl::nn::cuda::MM_BEST_FIT;
    engines_.emplace_back(ppl::nn::cuda::EngineFactory::Create(options));

    bool import_algo = false;

#if PPL_CUDA_IMPORT_FROM_BUFFER
    auto algo = model.ReadFile(config.weights);
    if (algo) {
      auto ret =
          engines_.back()->Configure(ppl::nn::cuda::ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER,
                                     algo.value().c_str(), algo.value().size());
      if (ret == ppl::common::RC_SUCCESS) {
        import_algo = true;
      } else {
        MMDEPLOY_ERROR("failed to import algorithms ({}), default algorithms will be used", ret);
      }
    }
#endif

    if (!import_algo) {
      engines_.back()->Configure(ppl::nn::cuda::ENGINE_CONF_USE_DEFAULT_ALGORITHMS, true);
    }
  }
#endif
#if PPL_NN_HAS_X86
  if (device_.is_host()) {
    ppl::nn::x86::RegisterBuiltinOpImpls();
    engines_.emplace_back(ppl::nn::x86::EngineFactory::Create({}));
  }
#endif
#if PPL_NN_HAS_RISCV
  if (device_.is_host()) {
    ppl::nn::riscv::RegisterBuiltinOpImpls();
    ppl::nn::riscv::EngineOptions options{};
    // TODO:
    //   FP16 -> postprocess
    options.forward_precision = ppl::common::DATATYPE_FLOAT32;
    options.dynamic_tuning_level = 0;
    options.winograd_level = 1;
    engines_.emplace_back(ppl::nn::riscv::EngineFactory::Create(options));
  }
#endif

  std::vector<ppl::nn::Engine*> engines;
  for (const auto& engine : engines_) {
    engines.push_back(engine.get());
  }

  OUTCOME_TRY(auto builder, ppl_try(ppl::nn::onnx::RuntimeBuilderFactory::Create()));
  OUTCOME_TRY(ppl_try(builder->LoadModel(onnx.data(), onnx.size(), nullptr)));

  ppl::nn::onnx::RuntimeBuilder::Resources resources{};
  resources.engines = engines.data();
  resources.engine_num = engines.size();
  OUTCOME_TRY(ppl_try(builder->SetResources(resources)));
  OUTCOME_TRY(ppl_try(builder->Preprocess()));

  OUTCOME_TRY(auto runtime, ppl_try(builder->CreateRuntime()));

  for (int i = 0; i < runtime->GetInputCount(); ++i) {
    auto src = runtime->GetInputTensor(i);
    inputs_internal_.push_back(src);
    inputs_external_.push_back(CreateInternalTensor(src, device_));

    /// debug only
    const auto& desc = *inputs_internal_[i]->GetShape();
    std::vector<long> shape_(desc.GetDims(), desc.GetDims() + desc.GetDimCount());
    MMDEPLOY_DEBUG("input {}: datatype = {}, dataformat = {}, shape = {}", i,
                   ppl::common::GetDataTypeStr(desc.GetDataType()),
                   ppl::common::GetDataFormatStr(desc.GetDataFormat()), shape_);
  }

  for (int i = 0; i < runtime->GetOutputCount(); ++i) {
    auto src = runtime->GetOutputTensor(i);
    outputs_internal_.push_back(src);
    outputs_external_.push_back(CreateInternalTensor(src, device_));

    const auto& desc = *outputs_internal_[i]->GetShape();
    std::vector<long> shape_(desc.GetDims(), desc.GetDims() + desc.GetDimCount());
    MMDEPLOY_DEBUG("output {}: datatype = {}, dataformat = {}, shape = {}", i,
                   ppl::common::GetDataTypeStr(desc.GetDataType()),
                   ppl::common::GetDataFormatStr(desc.GetDataFormat()), shape_);
    TensorShape shape(desc.GetDims(), desc.GetDims() + desc.GetDimCount());
  }

  auto input_shapes = GetShapes(inputs_external_);
  if (auto input_batch_size = GetBatchSize(input_shapes)) {
    auto output_shapes = GetShapes(outputs_external_);
    if (auto output_batch_size = GetBatchSize(output_shapes)) {
      if (input_batch_size.value() == output_batch_size.value()) {
        can_infer_output_shapes_ = true;
      }
    }
  }

  runtime_ = std::move(runtime);
  return success();
}

Result<void> PPLNet::Deinit() {
  try {
    runtime_.reset();
    return success();
  } catch (...) {
    return Status(eFail);
  }
}

static TensorShape GetShape(const PPLTensor& tensor) {
  const auto& desc = *tensor.GetShape();
  return {desc.GetDims(), desc.GetDims() + desc.GetDimCount()};
}

Result<ppl::common::datatype_t> GetPPLDataType(DataType data_type) {
  switch (data_type) {
    case DataType::kFLOAT:
      return ppl::common::DATATYPE_FLOAT32;
    case DataType::kHALF:
      return ppl::common::DATATYPE_FLOAT16;
    case DataType::kINT8:
      return ppl::common::DATATYPE_INT8;
    case DataType::kINT32:
      return ppl::common::DATATYPE_INT32;
    case DataType::kINT64:
      return ppl::common::DATATYPE_INT64;
    default:
      return Status(eNotSupported);
  }
}

Result<DataType> GetMMDeployDataType(ppl::common::datatype_t data_type) {
  switch (data_type) {
    case ppl::common::DATATYPE_FLOAT32:
      return DataType::kFLOAT;
    case ppl::common::DATATYPE_FLOAT16:
      return DataType::kHALF;
    case ppl::common::DATATYPE_INT8:
      return DataType::kINT8;
    case ppl::common::DATATYPE_INT32:
      return DataType::kINT32;
    case ppl::common::DATATYPE_INT64:
      return DataType::kINT64;
    default:
      return Status(eNotSupported);
  }
}

Result<void> PPLNet::Forward() {
  OUTCOME_TRY(stream_.Wait());

  OUTCOME_TRY(ppl_try(runtime_->Run()));

  for (int i = 0; i < outputs_external_.size(); ++i) {
    auto& internal = *outputs_internal_[i];
    auto format = internal.GetShape()->GetDataFormat();
    if (format != ppl::common::DATAFORMAT_NDARRAY) {
      MMDEPLOY_ERROR("output {}'s format is {}, only NDARRAY is currently supported", i,
                     ppl::common::GetDataFormatStr(format));
      return Status(eNotSupported);
    }
    auto& external = outputs_external_[i];
    auto dtype_int = internal.GetShape()->GetDataType();
    OUTCOME_TRY(auto dtype_ext, GetPPLDataType(external.data_type()));
    auto shape_int = GetShape(internal);
    auto shape_ext = external.shape();
    auto data_int = internal.GetBufferPtr();
    auto data_ext = external.data();
    if (shape_int != shape_ext || dtype_int != dtype_ext || data_int != data_ext) {
      if (dtype_int != dtype_ext) {
        auto desc = external.desc();
        desc.shape = shape_int;
        OUTCOME_TRY(desc.data_type, GetMMDeployDataType(dtype_int));
        external = Tensor(desc, external.allocator());
      } else {
        external.Reshape(shape_int);
      }
      std::shared_ptr<void> data(data_int, [](void*) {});
      if (external.size() > 0) {
        OUTCOME_TRY(Tensor(external.desc(), data).CopyTo(external, stream_));
      } else {
        MMDEPLOY_WARN("copy skipped due to zero sized tensor: {} {}", external.name(),
                      external.shape());
      }
    }
  }

  OUTCOME_TRY(stream_.Wait());
  return success();
}

Result<void> PPLNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

Result<void> ReshapeLike(PPLTensor& dst, Tensor& src) {
  auto& dst_desc = *dst.GetShape();
  auto& src_desc = src.desc();
  OUTCOME_TRY(auto data_type, GetPPLDataType(src_desc.data_type));
  dst_desc.SetDataType(data_type);
  dst_desc.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
  dst_desc.Reshape({begin(src_desc.shape), end(src_desc.shape)});
  dst.SetBufferPtr(src.data());
  return success();
}

Result<void> PPLNet::Reshape(Span<TensorShape> input_shapes) {
  auto prev_in_shapes = GetShapes(inputs_external_);
  auto prev_out_shapes = GetShapes(outputs_external_);

  for (int i = 0; i < inputs_external_.size(); ++i) {
    auto& input = inputs_external_[i];
    input.Reshape(input_shapes[i]);
    OUTCOME_TRY(ReshapeLike(*inputs_internal_[i], input));
  }

  if (can_infer_output_shapes_) {
    OUTCOME_TRY(auto output_shapes,
                InferOutputShapes(input_shapes, prev_in_shapes, prev_out_shapes));
    MMDEPLOY_DEBUG("inferred output shapes: {}", output_shapes);
    for (int i = 0; i < outputs_external_.size(); ++i) {
      auto& output = outputs_external_[i];
      output.Reshape(output_shapes[i]);
      OUTCOME_TRY(ReshapeLike(*outputs_internal_[i], output));
    }
  }
  return success();
}

Result<Span<Tensor>> PPLNet::GetInputTensors() { return inputs_external_; }

Result<Span<Tensor>> PPLNet::GetOutputTensors() { return outputs_external_; }

std::vector<TensorShape> PPLNet::GetShapes(Span<Tensor> tensors) {
  std::vector<TensorShape> shapes;
  shapes.reserve(tensors.size());
  for (const auto& t : tensors) {
    shapes.push_back(t.shape());
  }
  return shapes;
}

Result<int64_t> PPLNet::GetBatchSize(Span<TensorShape> shapes) {
  int64_t batch_size = -1;
  for (const auto& s : shapes) {
    if (s.empty()) {
      return Status(eNotSupported);
    }
    if (batch_size < 0) {
      batch_size = s.front();
    } else if (batch_size != s.front()) {
      return Status(eNotSupported);
    }
  }
  return batch_size;
}

Result<std::vector<TensorShape>> PPLNet::InferOutputShapes(Span<TensorShape> input_shapes,
                                                           Span<TensorShape> prev_in_shapes,
                                                           Span<TensorShape> prev_out_shapes) {
  OUTCOME_TRY(auto batch_size, GetBatchSize(input_shapes));
  if (input_shapes.size() != prev_in_shapes.size()) {
    return Status(eInvalidArgument);
  }
  for (int i = 0; i < input_shapes.size(); ++i) {
    prev_in_shapes[i][0] = batch_size;
    if (prev_in_shapes[i] != input_shapes[i]) {
      return Status(eNotSupported);
    }
  }
  std::vector<TensorShape> output_shapes(prev_out_shapes.begin(), prev_out_shapes.end());
  for (auto& shape : output_shapes) {
    shape[0] = batch_size;
  }
  return output_shapes;
}

PPLNet::~PPLNet() = default;

class PPLNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "pplnn"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    auto p = std::make_unique<PPLNet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating PPLNet: {}", r.error().message().c_str());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, PPLNetCreator);

}  // namespace mmdeploy
