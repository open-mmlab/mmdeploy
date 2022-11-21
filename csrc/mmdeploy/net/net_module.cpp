// Copyright (c) OpenMMLab. All rights reserved.

#include "net_module.h"

#include <thread>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/net.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/utils/scope_counter.h"
#include "mmdeploy/experimental/module_adapter.h"

using std::string;
using std::vector;

namespace mmdeploy::framework {

struct NetModule::Impl {
  using Input = std::map<std::string, Tensor>;
  using Output = std::map<std::string, Tensor>;

  explicit Impl(const Value& args) {
    MMDEPLOY_DEBUG("Net Module cfg: {}", args);
    auto init = [&]() -> Result<void> {
      auto name = args["name"].get<std::string>();
      auto& context = args["context"];
      if (context.contains("scope")) {
        is_profiling_ = true;
      }
      auto model = context["model"].get<Model>();
      OUTCOME_TRY(auto config, model.GetModelConfig(name));
      device_ = context.value("device", Device{"cpu"});
      stream_ = context.value("stream", Stream::GetDefault(device_));
      auto creator = gRegistry<Net>().Get(config.backend);
      if (!creator) {
        MMDEPLOY_ERROR("Net backend not found: {}, available backends: {}", config.backend,
                       gRegistry<Net>().List());
        return Status(eEntryNotFound);
      }
      auto net_cfg = args;
      net_cfg["context"].update({{"device", device_}, {"stream", stream_}});
      net_ = creator->Create(net_cfg);
      if (!net_) {
        MMDEPLOY_ERROR("Failed to create Net backend: {}, config: {}", config.backend, net_cfg);
        return Status(eFail);
      }
      OUTCOME_TRY(InitializeInputTensors(args));
      OUTCOME_TRY(InitializeOutputTensors(args));
      return success();
    };
    init().value();
  }

  Result<void> InitializeInputTensors(const Value& args) {
    auto inputs = args.value<Value>("input_map", ValueType::kObject);
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
      input_mapping_.insert({(*it).get<std::string>(), it.key()});
    }
    OUTCOME_TRY(inputs_, net_->GetInputTensors());
    for (const auto& t : inputs_) {
      input_mapping_.insert({t.name(), t.name()});
    }
    return success();
  }

  Result<void> InitializeOutputTensors(const Value& args) {
    auto outputs = args.value<Value>("output_map", ValueType::kObject);
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      output_mapping_.insert({(*it).get<std::string>(), it.key()});
    }
    OUTCOME_TRY(outputs_, net_->GetOutputTensors());
    for (const auto& t : outputs_) {
      output_mapping_.insert({t.name(), t.name()});
    }
    return success();
  }

  Result<TensorShape> InferInputShape(const vector<Tensor>& input) {
    auto batch_size = input.size();
    auto& exemplar = input.front();
    auto shape = exemplar.shape();
    if (batch_size == 1) {
      return shape;
    }
    if (shape[0] != 1) {
      MMDEPLOY_ERROR("unsupported shape for batch assemble: {}", shape);
      return Status(eNotSupported);
    }
    for (int i = 1; i < input.size(); ++i) {
      auto& sample = input[i];
      if (sample.shape() != shape) {
        MMDEPLOY_ERROR("shapes are not consistent across the batch");
        return Status(eNotSupported);
      }
    }
    shape[0] = static_cast<int64_t>(batch_size);
    return shape;
  }

  Result<vector<TensorShape> > InferInputShape(const vector<vector<Tensor> >& inputs) {
    vector<TensorShape> shapes;
    shapes.reserve(inputs.size());
    for (const auto& input : inputs) {
      OUTCOME_TRY(auto shape, InferInputShape(input));
      shapes.push_back(std::move(shape));
    }
    return shapes;
  }

  Result<std::vector<Output> > Forward(const std::vector<Input>& input) {
    //    auto t0 = std::chrono::high_resolution_clock::now();
    //
    auto batch_size = static_cast<int>(input.size());

    std::vector<std::vector<Tensor> > input_samples;
    input_samples.reserve(inputs_.size());
    for (const auto& t : inputs_) {
      auto name = input_mapping_.at(t.name());
      std::vector<Tensor> tmp;
      tmp.reserve(input.size());
      for (int i = 0; i < input.size(); ++i) {
        auto& sample = input[i];
        if (auto it = sample.find(name); it != sample.end()) {
          tmp.push_back(it->second);
        } else {
          MMDEPLOY_ERROR("sample {} missing key {}", i, name);
          return Status(eInvalidArgument);
        }
      }
      input_samples.push_back(std::move(tmp));
    }

    // 1. calculate input shape
    OUTCOME_TRY(auto input_shapes, InferInputShape(input_samples));

    // 2. call backend's reshape
    OUTCOME_TRY(net_->Reshape(input_shapes));

    // 3. fill input tensor
    for (int i = 0; i < inputs_.size(); ++i) {
      auto& src = input_samples[i];
      auto& dst = inputs_[i];
      if (dst.shape() != input_shapes[i]) {
        MMDEPLOY_ERROR("inconsistent input shape, expect {}, got {}", input_shapes[i], dst.shape());
        return Status(eFail);
      }
      if (src.size() > 1) {
        for (int j = 0; j < src.size(); ++j) {
          auto slice = dst.Slice(j);
          OUTCOME_TRY(src[j].CopyTo(slice, stream_));
        }
      } else {
        OUTCOME_TRY(src[0].CopyTo(dst, stream_));
      }
    }

    // 5. forward
    OUTCOME_TRY(net_->Forward());

    vector<Output> output(batch_size);
    for (const auto& t : outputs_) {
      auto name = output_mapping_.at(t.name());
      auto desc = t.desc();
      desc.device = device_;
      Tensor tmp(desc);
      if (tmp.size()) {
        OUTCOME_TRY(t.CopyTo(tmp, stream_));
      } else {
        MMDEPLOY_WARN("copy skipped due to zero sized tensor");
      }
      if (output.size() > 1) {
        for (int i = 0; i < output.size(); ++i) {
          output[i].emplace(name, tmp.Slice(i));
        }
      } else {
        output[0].emplace(name, std::move(tmp));
      }
    }
    if (is_profiling_) {
      OUTCOME_TRY(stream_.Wait());
    }

    return output;
  }

  Device device_;
  Stream stream_;
  std::unique_ptr<Net> net_;
  Span<Tensor> inputs_;
  Span<Tensor> outputs_;
  // outer scope to model input names
  std::map<std::string, std::string> input_mapping_;
  // outer scope to model output names
  std::map<std::string, std::string> output_mapping_;
  bool is_profiling_{false};
};

NetModule::~NetModule() = default;

NetModule::NetModule(NetModule&&) noexcept = default;

NetModule::NetModule(const Value& args) : impl_(std::make_unique<Impl>(args)) {}

Result<Value> NetModule::operator()(const Value& input) {
  auto filter = [](const Value& sample) {
    Impl::Input tensors;
    for (auto it = sample.begin(); it != sample.end(); ++it) {
      if (it->is_any<Tensor>()) {
        tensors.insert({it.key(), it->get<Tensor>()});
      }
    }
    return tensors;
  };
  std::vector<Impl::Input> batch;
  if (input.is_array()) {
    batch.reserve(input.size());
    for (const auto& sample : input) {
      batch.push_back(filter(sample));
    }
  } else if (input.is_object()) {
    batch.push_back(filter(input));
  } else {
    return Status(eNotSupported);
  }
  OUTCOME_TRY(auto batch_output, impl_->Forward(batch));
  if (input.is_array()) {
    return to_value(batch_output);
  } else {
    return to_value(batch_output.at(0));
  }
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (Net, 0),
                               [](const Value& config) { return CreateTask(NetModule{config}); });

}  // namespace mmdeploy::framework
