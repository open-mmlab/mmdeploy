// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/net/net_module.h"

#include <algorithm>
#include <numeric>
#include <thread>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/net.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
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
      for (const auto& meta : model.meta().models) {
        if (meta.name == name) {
          max_batch_size_ = meta.batch_size;
        }
      }
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

  static Result<TensorShape> InferBatchShape(const vector<Tensor>& input) {
    auto batch_size = input.size();
    auto& exemplar = input.front();
    auto shape = exemplar.shape();
    if (batch_size == 1) {
      return shape;
    }
    if (shape[0] != 1) {
      MMDEPLOY_WARN("unsupported shape for batch assemble: {}", shape);
      return Status(eNotSupported);
    }
    for (int i = 1; i < input.size(); ++i) {
      auto& sample = input[i];
      if (sample.shape() != shape) {
        MMDEPLOY_WARN("shapes are not consistent across the batch");
        return Status(eNotSupported);
      }
    }
    shape[0] = static_cast<int64_t>(batch_size);
    return shape;
  }

  static Result<vector<TensorShape>> InferBatchShape(const vector<vector<Tensor>>& inputs) {
    vector<TensorShape> shapes;
    shapes.reserve(inputs.size());
    for (const auto& input : inputs) {
      OUTCOME_TRY(auto shape, InferBatchShape(input));
      shapes.push_back(std::move(shape));
    }
    return shapes;
  }

  Result<vector<vector<Tensor>>> CollectInputTensors(const vector<Input>& inputs) {
    vector<vector<Tensor>> input_samples;
    input_samples.reserve(inputs_.size());
    for (const auto& t : inputs_) {
      auto name = input_mapping_.at(t.name());
      auto& tmp = input_samples.emplace_back();
      for (const auto& sample : inputs) {
        if (auto it = sample.find(name); it != sample.end()) {
          tmp.push_back(it->second);
        } else {
          MMDEPLOY_ERROR("sample {} missing key {}", &sample - inputs.data(), name);
          return Status(eInvalidArgument);
        }
      }
    }
    return input_samples;
  }

  void SaveBatch(vector<vector<Tensor>> samples, vector<int> indices,
                 vector<vector<vector<Tensor>>>& batch_tensors,
                 vector<vector<TensorShape>>& batch_shapes,
                 vector<vector<int>>& batch_sample_idxs) const {
    if (auto maybe_batch_shape = InferBatchShape(samples)) {
      batch_shapes.push_back(maybe_batch_shape.value());
      batch_tensors.push_back(std::move(samples));
      batch_sample_idxs.push_back(std::move(indices));
    } else {
      // cannot assemble batch, do it one by one
      for (int k = 0; k < indices.size(); ++k) {
        auto& shapes = batch_shapes.emplace_back();
        auto& batch = batch_tensors.emplace_back(inputs_.size());
        batch_sample_idxs.push_back({indices[k]});
        for (int j = 0; j < inputs_.size(); ++j) {
          shapes.push_back(samples[j][k].shape());
          batch[j].push_back(std::move(samples[j][k]));
        }
      }
    }
  }

  void SamplesToBatches(const vector<vector<Tensor>>& input_samples, size_t n_samples,
                        vector<vector<vector<Tensor>>>& batch_tensors,
                        vector<vector<TensorShape>>& batch_shapes,
                        vector<vector<int>>& batch_sample_idxs) const {
    // concat all shapes in samples to make comparison easier
    vector<vector<int64_t>> concat_shapes;
    concat_shapes.reserve(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
      auto& shape = concat_shapes.emplace_back();
      for (const auto& input : input_samples) {
        shape.insert(shape.end(), input[i].shape().begin(), input[i].shape().end());
      }
    }

    // cluster samples by concatenated shapes
    vector<int> shape_idxs(concat_shapes.size());
    std::iota(shape_idxs.begin(), shape_idxs.end(), 0);
    std::sort(shape_idxs.begin(), shape_idxs.end(),
              [&concat_shapes](int i, int j) { return concat_shapes[i] < concat_shapes[j]; });
    shape_idxs.erase(std::unique(shape_idxs.begin(), shape_idxs.end(),
                                 [&concat_shapes](int i, int j) {
                                   return concat_shapes[i] == concat_shapes[j];
                                 }),
                     shape_idxs.end());

    // generate batches of samples with equal shapes, limit the batch size by max_batch_size_
    for (const auto ref_shape_idx : shape_idxs) {
      const auto& ref_shape = concat_shapes[ref_shape_idx];
      vector<vector<Tensor>> samples(inputs_.size());
      vector<int> indices;
      for (size_t i = 0; i < concat_shapes.size(); ++i) {
        if (concat_shapes[i] == ref_shape) {
          for (size_t j = 0; j < inputs_.size(); ++j) {
            samples[j].push_back(input_samples[j][i]);
          }
          indices.push_back(static_cast<int>(i));
          if (indices.size() == max_batch_size_) {
            SaveBatch(std::move(samples), std::move(indices), batch_tensors, batch_shapes,
                      batch_sample_idxs);
            samples = vector<vector<Tensor>>(inputs_.size());
            indices = {};
          }
        }
      }
      if (!indices.empty()) {
        SaveBatch(std::move(samples), std::move(indices), batch_tensors, batch_shapes,
                  batch_sample_idxs);
      }
    }
  }

  Result<vector<Output>> Forward(const vector<Input>& inputs) {
    OUTCOME_TRY(auto input_samples, CollectInputTensors(inputs));

    vector<vector<vector<Tensor>>> batch_tensors;
    vector<vector<TensorShape>> batch_shapes;
    vector<vector<int>> batch_sample_indices;

    SamplesToBatches(input_samples, inputs.size(), batch_tensors, batch_shapes,
                     batch_sample_indices);

    vector<Output> outputs(inputs.size());
    for (size_t i = 0; i < batch_tensors.size(); ++i) {
      OUTCOME_TRY(net_->Reshape(batch_shapes[i]));
      OUTCOME_TRY(CopyInputTensors(batch_tensors[i], batch_shapes[i]));
      OUTCOME_TRY(net_->Forward());
      OUTCOME_TRY(CopyOutputTensors(batch_sample_indices[i], outputs));
      if (i + 1 < batch_tensors.size()) {  // sync if not the last batch
        OUTCOME_TRY(stream_.Wait());
      }
    }

    if (is_profiling_) {
      OUTCOME_TRY(stream_.Wait());
    }

    return outputs;
  }

  Result<void> CopyInputTensors(const vector<vector<Tensor>>& batch,
                                const vector<TensorShape>& shapes) const {
    for (int i = 0; i < inputs_.size(); ++i) {
      auto& src = batch[i];
      auto& dst = inputs_[i];
      if (dst.shape() != shapes[i]) {
        MMDEPLOY_ERROR("inconsistent input shape, expect {}, got {}", shapes[i], dst.shape());
        return Status(eFail);
      }
      if (src.size() > 1) {
        for (int j = 0; j < src.size(); ++j) {
          OUTCOME_TRY(dst.Slice(j).CopyFrom(src[j], stream_));
        }
      } else {
        OUTCOME_TRY(src.front().CopyTo(dst, stream_));
      }
    }
    return success();
  }

  Result<void> CopyOutputTensors(const vector<int>& indices, vector<Output>& outputs) {
    for (const auto& output : outputs_) {
      auto name = output_mapping_.at(output.name());
      auto desc = output.desc();
      desc.device = device_;
      Tensor tmp(desc);
      if (tmp.size()) {
        OUTCOME_TRY(output.CopyTo(tmp, stream_));
      } else {
        MMDEPLOY_WARN("copy skipped due to zero sized tensor");
      }
      if (indices.size() > 1) {
        for (int i = 0; i < indices.size(); ++i) {
          outputs[indices[i]].emplace(name, tmp.Slice(i));
        }
      } else {
        outputs[indices.front()].emplace(name, std::move(tmp));
      }
    }
    return success();
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
  int max_batch_size_{1};
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
