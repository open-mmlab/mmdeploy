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
    // tag_invoke(Transfer, std::move(input), *sched_);
    return LetValue(std::move(input), [this](Value& v) -> Sender<Value> {
      //      MMDEPLOY_INFO("name = {}, val = {}", name(), v);
      if (v.front().is_array() && !is_batched_) {
        // clang-format off
        auto batch_size = v.front().size();
        Value output = Value::Array(batch_size);
        return Just(std::move(output))
             | Then([&](Value&& output) -> Value {
                   auto input = graph::DistribAA(v).value();
                   return Value{std::move(input), std::move(output)};
                 })
             | Transfer(*sched_)
             | TypeErase()
             | Bulk(batch_size, [&](size_t index, Value& in_out) {
                   const auto& input = in_out[0];
                   auto& output = in_out[1];
                   output[index] = module_->Process(input[index]).value();
                 })
             | Then([](const Value& in_out) {
                   return graph::DistribAA(in_out[1]).value();
                 });
        // clang-format on
      } else {
        auto output = module_->Process(v).value();
        return Just(std::move(output)) | Transfer(*sched_);
      }
    });
  }

 private:
  std::optional<TypeErasedScheduler<Value>> sched_;
  unique_ptr<Module> module_;
  bool is_batched_{false};
  bool is_thread_safe_{false};
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
