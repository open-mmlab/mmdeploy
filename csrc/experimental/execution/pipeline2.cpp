// Copyright (c) OpenMMLab. All rights reserved.

#include "pipeline2.h"

#include "archive/value_archive.h"
#include "deferred_batch_operation.h"
#include "graph/common.h"
#include "inlined_scheduler.h"

namespace mmdeploy {

namespace async {

struct Pipeline::State {
 public:
  State(vector<int> use_count, Sender<Value> args);

  void Write(int index, Sender<Value> value);
  // ! coords must last until finish of the async operation.
  Sender<Value> Collect(const vector<Coords>& coords);

 private:
  Sender<Value> Read(int index);
  // collect inputs from outputs of multiple nodes
  Sender<Value> CollectN(const vector<Coords>& coords);
  // collect inputs from 1 node's outputs
  Sender<Value> Collect1(const Coords& coords);

 private:
  vector<int> use_count_;
  std::vector<std::optional<Sender<Value>>> values_;
};

Sender<Value> Pipeline::State::CollectN(const vector<Coords>& coords) {
  vector<Sender<Value>> predecessors;
  predecessors.reserve(coords.size());
  size_t count = 0;
  for (const auto& coord : coords) {
    predecessors.push_back(Read(coord.index));
    count += coord.mapping.size();
  }
  return Then(WhenAll_(std::move(predecessors)), [count, &coords](Value::Array vals) {
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
  assert(!values_[index]);
  if (use_count_[index] > 1) {
    // ! split to create a copyable sender
    values_[index] = Split(std::move(value));
  } else {
    values_[index] = std::move(value);
  }
}

Sender<Value> Pipeline::State::Read(int index) {
  assert(values_[index]);
  if (--use_count_[index] == 0) {
    return std::move(*values_[index]);
  } else {
    // ! copy ctor of the wrapped sender must be valid
    return *values_[index];
  }
}

Pipeline::State::State(vector<int> use_count, Sender<Value> args)
    : use_count_(std::move(use_count)), values_(use_count_.size()) {
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

/////////////////////////////////////////////////////////////////////
/// parsers

using graph::CreateFromRegistry;

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
    OUTCOME_TRY(task->module_, CreateFromRegistry<Module>(config, "module"));
    if (config.contains("scheduler")) {
      auto sched_name = config["scheduler"].get<string>();
      task->sched_ = config["context"]["schedulers"][sched_name].get<TypeErasedScheduler<Value>>();
    } else {
      task->sched_ = TypeErasedScheduler<Value>{InlineScheduler{}};
    }
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

    Value schedulers(Value::kObject);
    if (config.contains("create_scheduler")) {
      const auto& sched_cfg = config["create_scheduler"];
      for (auto it = sched_cfg.begin(); it != sched_cfg.end(); ++it) {
        OUTCOME_TRY(schedulers[it.key()], CreateFromRegistry<TypeErasedScheduler<Value>>(*it));
      }
    }

    const auto& task_configs = config["pipeline"]["tasks"];
    auto size = task_configs.size();

    vector<unique_ptr<Node>> nodes;
    nodes.reserve(size);

    vector<vector<Pipeline::Coords>> input_coords;
    input_coords.reserve(size);

    use_count_.resize(size + 1);

    // MMDEPLOY_INFO("pipeline->inputs: {}", pipeline->inputs());
    OUTCOME_TRY(UpdateOutputCoords(static_cast<int>(size), pipeline->inputs()));
    for (auto task_config : task_configs) {
      auto index = static_cast<int>(nodes.size());

      auto name = task_config.value<string>("name", "");
      auto type = task_config.value<string>("type", "");
      // propagate context
      if (config.contains("context")) {
        task_config["context"].update(config["context"]);
        if (!schedulers.empty()) {
          task_config["context"]["schedulers"].update(schedulers);
        }
      }
      OUTCOME_TRY(auto node, CreateFromRegistry<Node>(task_config));
      if (node) {
        //        if (node->name() == "yolox") {
        //          node = std::make_unique<DeferredBatchOperation>(std::move(node), 1,
        //                                                          std::chrono::milliseconds(10000));
        //        }
        OUTCOME_TRY(auto coords, GetInputCoords(node->inputs()));
        input_coords.push_back(std::move(coords));
        OUTCOME_TRY(UpdateOutputCoords(index, node->outputs()));
        nodes.push_back(std::move(node));
      } else {
        MMDEPLOY_ERROR("could not create {}: {}", name, type);
        return Status(eFail);
      }
    }
    OUTCOME_TRY(auto coords, GetInputCoords(pipeline->outputs()));

    pipeline->nodes_ = std::move(nodes);
    pipeline->use_count_ = std::move(use_count_);
    pipeline->input_coords_ = std::move(input_coords);
    pipeline->ret_coords_ = std::move(coords);

    return std::move(pipeline);

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", e.what());
    return Status(eFail);
  }
}

Result<vector<Pipeline::Coords>> PipelineParser::GetInputCoords(const vector<string>& names) {
  // MMDEPLOY_INFO("GetInputCoords: {}", names);
  vector<Pipeline::Coords> ret;
  ret.reserve(names.size());
  for (int i = 0; i < names.size(); ++i) {
    const auto& input = names[i];
    if (auto it = output_name_to_coords_.find(input); it != output_name_to_coords_.end()) {
      const auto& [node_id, port_id] = it->second;
      ++use_count_[node_id];
      auto ct = find_if(begin(ret), end(ret),
                        [node_id = node_id](auto& c) { return c.index == node_id; });
      if (ct == end(ret)) {
        ct = ret.insert(ct, {node_id, {}});
      }
      ct->mapping.emplace_back(port_id, i);
    } else {
      MMDEPLOY_ERROR("missing input: {}", input);
      return Status(eEntryNotFound);
    }
  }
  return ret;
}

Result<void> PipelineParser::UpdateOutputCoords(int index, const vector<string>& names) {
  for (int i = 0; i < names.size(); ++i) {
    const auto& output = names[i];
    if (auto it = output_name_to_coords_.find(output); it != output_name_to_coords_.end()) {
      MMDEPLOY_ERROR("duplicate output: ", output);
      return Status(eNotSupported);
    } else {
      // MMDEPLOY_ERROR("insert: {}", output);
      output_name_to_coords_.insert({output, {index, i}});
    }
  }
  return success();
}

void __link_inference();
void __link_scheduler();

class PipelineCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Pipeline"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override {
    __link_inference();
    __link_scheduler();
    return PipelineParser{}.Parse(value).value();
  }
};

class TaskCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Task"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override {
    return TaskParser::Parse(value).value();
  }
};

REGISTER_MODULE(Node, TaskCreator);
REGISTER_MODULE(Node, PipelineCreator);

}  // namespace async

MMDEPLOY_DEFINE_REGISTRY(async::Node);

MMDEPLOY_DEFINE_REGISTRY(TypeErasedScheduler<Value>);

}  // namespace mmdeploy
