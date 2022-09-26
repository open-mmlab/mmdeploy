// Copyright (c) OpenMMLab. All rights reserved.

#include "ncnn_net.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "ncnn_ops_register.h"

namespace mmdeploy::framework {

NCNNNet::~NCNNNet() {}

Result<void> ncnn_status(int code) {
  if (code == 0) {
    return success();
  }
  return Status(eFail);
}

Result<void> NCNNNet::Init(const Value& args) {
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();
  if (args.contains("use_vulkan")) {
    net_.opt.use_vulkan_compute = args["use_vulkan"].get<bool>();
  }
  if (!device_.is_host()) {
    return Status(eNotSupported);
  }

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  auto precision = config.precision;
  if (precision == "FP16") {
    net_.opt.use_fp16_packed = true;
    net_.opt.use_fp16_storage = true;
    net_.opt.use_fp16_arithmetic = true;
  } else if (precision == "INT8") {
    // in android platform, ncnn will automatically start FP16 accelerate.
    // In INT8 case, we set fp16 as false explicitly.
    net_.opt.use_int8_packed = true;
    net_.opt.use_int8_storage = true;
    net_.opt.use_int8_arithmetic = true;
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
  } else {
    // in android platform, ncnn will automatically start FP16 accelerate.
    // In FP32 case, we set fp16 as false explicitly.
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
  }
  OUTCOME_TRY(params_, model.ReadFile(config.net));
  OUTCOME_TRY(weights_, model.ReadFile(config.weights));
  register_mmdeploy_custom_layers(net_);

  OUTCOME_TRY(ncnn_status(net_.load_param_mem(params_.c_str())));
  net_.load_model(reinterpret_cast<const unsigned char*>(weights_.data()));

  input_indices_ = net_.input_indexes();
  for (const auto& x : net_.input_names()) {
    input_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        DataType::kFLOAT,
        {},
        x,
    });
  }
  output_indices_ = net_.output_indexes();
  for (const auto& x : net_.output_names()) {
    output_tensors_.emplace_back(TensorDesc{
        Device("cpu"),
        DataType::kFLOAT,
        {},
        x,
    });
  }
  return success();
}

Result<void> NCNNNet::Deinit() { return success(); }

Result<void> NCNNNet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

Result<Span<Tensor>> NCNNNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> NCNNNet::GetOutputTensors() { return output_tensors_; }

// TODO: discuss a policy for batch processing
Result<void> NCNNNet::Forward() {
  auto extractor = net_.create_extractor();
  OUTCOME_TRY(stream_.Wait());
  std::vector<ncnn::Mat> inputs(input_indices_.size());
  for (size_t i = 0; i < input_indices_.size(); ++i) {
    auto& tensor = input_tensors_[i];
    auto shape = tensor.shape();
    assert(shape[0] == 1);
    inputs[i] = ncnn::Mat(shape[3], shape[2], shape[1], tensor.data());
    OUTCOME_TRY(ncnn_status(extractor.input(input_indices_[i], inputs[i])));
  }
  std::vector<ncnn::Mat> outputs(output_indices_.size());
  for (size_t i = 0; i < output_indices_.size(); ++i) {
    OUTCOME_TRY(ncnn_status(extractor.extract(output_indices_[i], outputs[i])));
    auto& tensor = output_tensors_[i];
    auto shape = outputs[i].shape();
    if (outputs[i].dims == 1) {
      tensor.Reshape({1, shape.w});
    } else if (outputs[i].dims == 2) {
      tensor.Reshape({1, shape.h, shape.w});
    } else if (outputs[i].dims == 3) {
      tensor.Reshape({1, shape.c, shape.h, shape.w});
    } else {
      // for dim==4 case and blank image.
      tensor.Reshape({1, shape.c, shape.d, shape.h, shape.w});
    }
    // tensor.Reshape({1, shape.c, shape.h, shape.w});
    // ncnn Mat may be padded, flatten to avoid that
    auto flattened = outputs[i].reshape(shape.c * shape.h * shape.w);
    // if ((shape.c * shape.h * shape.w) > 0)
    if (outputs[i].dims > 0) {
      OUTCOME_TRY(tensor.CopyFrom(flattened.data, stream_));
    }
    OUTCOME_TRY(stream_.Wait());
  }
  return success();
}

class NCNNNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "ncnn"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    auto p = std::make_unique<NCNNNet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating NCNNNet: {}", r.error().message().c_str());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, NCNNNetCreator);

}  // namespace mmdeploy::framework
