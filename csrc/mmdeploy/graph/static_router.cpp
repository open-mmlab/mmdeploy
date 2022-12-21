// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/static_router.h"

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/execution/schedulers/inlined_scheduler.h"
#include "mmdeploy/graph/common.h"

namespace mmdeploy::graph {

class StaticRouter::State {
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
  vector<std::optional<Sender<Value>>> values_;
};

Sender<Value> StaticRouter::State::CollectN(const vector<Coords>& coords) {
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

Sender<Value> StaticRouter::State::Collect1(const StaticRouter::Coords& coords) {
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

Sender<Value> StaticRouter::State::Collect(const vector<Coords>& coords) {
  if (coords.size() == 1) {
    return Collect1(coords[0]);
  } else {
    return CollectN(coords);
  }
}

void StaticRouter::State::Write(int index, Sender<Value> value) {
  assert(!values_[index]);
  if (use_count_[index] > 1) {
    // ! split to create a copyable sender
    values_[index] = Split(std::move(value));
  } else {
    values_[index] = std::move(value);
  }
}

Sender<Value> StaticRouter::State::Read(int index) {
  assert(values_[index]);
  if (--use_count_[index] == 0) {
    return std::move(*values_[index]);
  } else {
    // ! copy ctor of the wrapped sender must be valid
    return *values_[index];
  }
}

StaticRouter::State::State(vector<int> use_count, Sender<Value> args)
    : use_count_(std::move(use_count)), values_(use_count_.size()) {
  values_.back() = std::move(args);
}

Sender<Value> StaticRouter::Process(Sender<Value> args) {
  auto index = std::make_shared<profiler::Index>();
  auto start = std::make_shared<bool>(false);
  if (scope_) {
    *index = scope_->next_.fetch_add(1, std::memory_order_relaxed);
    args = Then(std::move(args), [this, index, start](Value v) mutable {
      if (*start == false) {
        scope_->Add(profiler::Event::kStart, *index, profiler::Clock::now());
        *start = true;
      }
      return std::move(v);
    });
  }

  State state(use_count_, std::move(args));
  for (size_t i = 0; i < nodes_.size(); ++i) {
    auto input = state.Collect(input_coords_[i]);
    auto output = nodes_[i]->Process(std::move(input));
    state.Write(static_cast<int>(i), std::move(output));
  }
  auto output = state.Collect(ret_coords_);
  if (scope_) {
    output = Then(std::move(output), [this, index](Value v) {
      scope_->Add(profiler::Event::kEnd, *index, profiler::Clock::now());
      return std::move(v);
    });
  }
  return output;
}

/////////////////////////////////////////////////////////////////////
/// parsers

Result<unique_ptr<StaticRouter>> StaticRouterBuilder::Build(const Value& config) {
  try {
    auto pipeline = std::make_unique<StaticRouter>();
    if (config.contains("context") && config["context"].contains("scope")) {
      auto name = config.value("name", std::string("Pipeline"));
      auto scope = config["context"]["scope"].get<profiler::Scope*>();
      pipeline->scope_ = scope->CreateScope(name);
    }

    const auto& task_configs = config["tasks"];
    auto size = task_configs.size();

    vector<unique_ptr<Node>> nodes;
    nodes.reserve(size);

    vector<vector<StaticRouter::Coords>> input_coords;
    input_coords.reserve(size);

    use_count_.resize(size + 1);

    OUTCOME_TRY(auto inputs, ParseStringArray(config["input"]));
    OUTCOME_TRY(auto outputs, ParseStringArray(config["output"]));

    OUTCOME_TRY(UpdateOutputCoords(static_cast<int>(size), inputs));
    for (auto task_config : task_configs) {
      auto index = static_cast<int>(nodes.size());

      auto name = task_config.value<string>("name", "");
      auto type = task_config.value<string>("type", "");
      // propagate context
      if (!task_config.contains("context")) {
        task_config["context"] = Value::Object();
      }
      if (config.contains("context")) {
        update(task_config["context"].object(), config["context"].object(), 2);
        if (pipeline->scope_) {
          task_config["context"]["scope"] = pipeline->scope_;
        }
      }

      OUTCOME_TRY(auto builder, Builder::CreateFromConfig(task_config));
      if (builder) {
        auto node = builder->Build().value();
        OUTCOME_TRY(auto coords, GetInputCoords(builder->inputs()));
        input_coords.push_back(std::move(coords));
        OUTCOME_TRY(UpdateOutputCoords(index, builder->outputs()));
        nodes.push_back(std::move(node));
      } else {
        MMDEPLOY_ERROR("could not create {}: {}", name, type);
        return Status(eFail);
      }
    }
    OUTCOME_TRY(auto coords, GetInputCoords(outputs));

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

Result<vector<StaticRouter::Coords>> StaticRouterBuilder::GetInputCoords(
    const vector<string>& names) {
  vector<StaticRouter::Coords> ret;
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
      for (const auto& [k, v] : output_name_to_coords_) {
        MMDEPLOY_ERROR("local var: {}", k);
      }
      return Status(eEntryNotFound);
    }
  }
  return ret;
}

Result<void> StaticRouterBuilder::UpdateOutputCoords(int index, const vector<string>& names) {
  for (int i = 0; i < names.size(); ++i) {
    const auto& output = names[i];
    if (auto it = output_name_to_coords_.find(output); it != output_name_to_coords_.end()) {
      MMDEPLOY_ERROR("duplicate output: ", output);
      return Status(eNotSupported);
    } else {
      output_name_to_coords_.insert({output, {index, i}});
    }
  }
  return success();
}

}  // namespace mmdeploy::graph
