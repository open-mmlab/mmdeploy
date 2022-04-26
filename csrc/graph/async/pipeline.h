// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_

#include <map>

#include "core/module.h"
#include "core/operator.h"
#include "core/value.h"
#include "execution/schedulers/registry.h"
#include "execution/type_erased.h"
#include "execution/when_all_value.h"

namespace mmdeploy {

namespace async {

using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

template <class... Ts>
using Sender = TypeErasedSender<Ts...>;

class Node {
  friend class NodeParser;

 public:
  virtual ~Node() = default;
  virtual Sender<Value> Process(Sender<Value> input) = 0;
  const vector<string>& inputs() const noexcept { return inputs_; }
  const vector<string>& outputs() const noexcept { return outputs_; }
  const string& name() const noexcept { return name_; }

 protected:
  string name_;
  vector<string> inputs_;
  vector<string> outputs_;
};

class NodeParser {
 public:
  static Result<void> Parse(const Value& config, Node& node);
};

class Task : public Node {
  friend class TaskParser;

 public:
  Sender<Value> Process(Sender<Value> input) override {
    return LetValue(sched_->ScheduleFrom(std::move(input)), [this](Value& v) -> Sender<Value> {
      if (v[0].is_array()) {
        auto output = Then(Schedule(*sched_), [&]() -> Value { return Value::Array(v.size()); });
        auto process = sched_->Bulk(std::move(output), v.size(), [&](size_t index, Value& output) {
          output[index] = module_->Process(v).value();
        });
        return process;
      } else {
        auto output = module_->Process(v).value();
        return Just(std::move(output));
      }
    });
  }

 private:
  std::optional<TypeErasedScheduler<Value>> sched_;
  unique_ptr<Module> module_;
};

class TaskParser {
 public:
  static Result<unique_ptr<Task>> Parse(const Value& config);
};

class Pipeline : public Node {
  friend class PipelineParser;

 public:
  Sender<Value> Process(Sender<Value> args) override;

  struct Coords {
    // source node index
    int index;
    // source output port -> destination input port mapping
    vector<pair<int, int>> mapping;
  };

  class State;

 private:
  vector<unique_ptr<Node>> nodes_;
  vector<int> use_count_;
  vector<vector<Coords>> input_coords_;
  vector<Coords> ret_coords_;
};

class PipelineParser {
 public:
  Result<unique_ptr<Pipeline>> Parse(const Value& config);

 private:
  Result<vector<Pipeline::Coords>> GetInputCoords(const vector<string>& names);

  Result<void> UpdateOutputCoords(int index, const vector<string>& names);

  // use count for each node's output
  vector<int> use_count_;
  // name -> (node_id, port_id)
  std::map<string, pair<int, int>> output_name_to_coords_;
};

}  // namespace async

MMDEPLOY_DECLARE_REGISTRY(async::Node);

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_
