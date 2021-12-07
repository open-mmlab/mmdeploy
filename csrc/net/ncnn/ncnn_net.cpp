// Copyright (c) OpenMMLab. All rights reserved.

#include "ncnn_net.h"

#include "core/logger.h"
#include "core/model.h"
#include "core/utils/formatter.h"

namespace mmdeploy {

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

  if (!device_.is_host()) {
    return Status(eNotSupported);
  }

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));

  OUTCOME_TRY(params_, model.ReadFile(config.net));
  OUTCOME_TRY(weights_, model.ReadFile(config.weights));

  OUTCOME_TRY(ncnn_status(net_.load_param_mem(params_.c_str())));
  net_.load_model(reinterpret_cast<const unsigned char*>(weights_.data()));

  input_indices_ = net_.input_indexes();
  for (const auto& x : net_.input_names()) {
    //    input_names_.emplace_back(x);
    input_tensors_.emplace_back(TensorDesc{
        .device = Device("cpu"),
        .data_type = DataType::kFLOAT,
        .shape = {},
        .name = x,
    });
  }
  output_indices_ = net_.output_indexes();
  for (const auto& x : net_.output_names()) {
    //    output_names_.emplace_back(x);
    output_tensors_.emplace_back(TensorDesc{
        .device = Device("cpu"),
        .data_type = DataType::kFLOAT,
        .shape = {},
        .name = x,
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
    tensor.Reshape({1, shape.w, shape.h, shape.c});
    // ncnn Mat may be padded, flatten to avoid that
    auto flattened = outputs[i].reshape(shape.w * shape.h * shape.c);
    OUTCOME_TRY(tensor.CopyFrom(flattened.data, stream_));
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
      ERROR("error creating NCNNNet: {}", r.error().message().c_str());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, NCNNNetCreator);

}  // namespace mmdeploy
