// Copyright (c) OpenMMLab. All rights reserved.

#include "pipeline2.h"

#include "archive/value_archive.h"
#include "graph/common.h"

namespace mmdeploy::async {

Sender<Value> Pipeline::State::CollectN(const vector<Coords>& coords) {
  vector<Sender<Value>> predecessors;
  predecessors.reserve(coords.size());
  size_t count = 0;
  for (const auto& coord : coords) {
    predecessors.push_back(Read(coord.index));
    count += coord.mapping.size();
  }
  return Then(WhenAll(std::move(predecessors)), [count, &coords](Value::Array vals) {
    Value ret(ValueType::kArray);
    auto& args = ret.array();
    args.resize(count);
    for (int j = 0; j < coords.size(); ++j) {
      for (const auto& [from, to] : coords[j].mapping) {
        // ! from(s) must be unique to avoid trouble, should be enforced by parser
        args[to] = std::move(vals[j][from]);
      }
    }
    return ret;
  });
}

Sender<Value> Pipeline::State::Collect1(const Pipeline::Coords& coords) {
  return Then(Read(coords.index), [&coords](Value val) {
    Value ret(ValueType::kArray);
    auto& args = ret.array();
    args.resize(coords.mapping.size());
    for (const auto& [from, to] : coords.mapping) {
      // ! from(s) must be unique to avoid trouble, should be enforced by parser
      args[to] = std::move(val[from]);
    }
    return ret;
  });
}

Sender<Value> Pipeline::State::Collect(const vector<Coords>& coords) {
  if (coords.size() == 1) {
    return Collect1(coords[0]);
  } else {
    return CollectN(coords);
  }
}

void Pipeline::State::Write(int index, Sender<Value> value) {
  if (use_count_[index] > 1) {
    // ! split to create a copyable sender
    values_[index] = Split(std::move(value));
  } else {
    values_[index] = std::move(value);
  }
}

Sender<Value> Pipeline::State::Read(int index) {
  if (--use_count_[index] == 0) {
    return std::move(*values_[index]);
  } else {
    // ! copy ctor of the wrapped sender must be valid
    return *values_[index];
  }
}

Pipeline::State::State(vector<int> use_count, Sender<Value> args)
    : use_count_(std::move(use_count)), values_(use_count_.size() + 1) {
  values_.back() = std::move(args);
}

Sender<Value> Pipeline::Process(Sender<Value> args) {
  State state(use_count_, std::move(args));
  for (size_t i = 0; i < nodes_.size(); ++i) {
    auto input = state.Collect(input_coords_[i]);
    auto output = nodes_[i]->Process(std::move(input));
    state.Write(static_cast<int>(i), std::move(output));
  }
  return state.Collect(ret_coords_);
}

Result<void> NodeParser::Parse(const Value& config, Node& node) {
  try {
    from_value(config["input"], node.inputs_);
    from_value(config["output"], node.outputs_);
    node.name_ = config.value<std::string>("name", "");
    return success();
  } catch (const Exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", config);
    return failure(e.code());
  }
}

Result<unique_ptr<Task>> TaskParser::Parse(const Value& config) {
  try {
    auto task = std::make_unique<Task>();
    OUTCOME_TRY(NodeParser::Parse(config, *task));
    OUTCOME_TRY(task->module_, graph::CreateFromRegistry<Module>(config, "module"));
    return std::move(task);
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", config);
    return nullptr;
  }
}

Result<unique_ptr<Pipeline>> PipelineParser::Parse(const Value& config) {
  try {
    auto pipeline = std::make_unique<Pipeline>();
    OUTCOME_TRY(NodeParser::Parse(config["pipeline"], *pipeline));
    for (auto task_config : config["pipeline"]["tasks"]) {

    }
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", e.what());
  }
}

}  // namespace mmdeploy::async
