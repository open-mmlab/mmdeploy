// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_

#include <map>

#include "core/module.h"
#include "core/operator.h"
#include "core/value.h"
#include "experimental/execution/type_erased.h"
#include "experimental/execution/when_all_value.h"

namespace mmdeploy::async {

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
    return Then(std::move(input), [&](const Value& v) { return module_->Process(v).value(); });
  }

 private:
  unique_ptr<Module> module_;
};

class TaskParser {
 public:
  Result<unique_ptr<Task>> Parse(const Value& config);
};

class Pipeline : public Node {
  friend class PipelineParser;

 public:
  Sender<Value> Process(Sender<Value> args) override;

  struct Coords {
    int index;
    std::vector<std::pair<int, int>> mapping;
  };

  class State {
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

 private:
  const vector<unique_ptr<Node>> nodes_;
  const vector<int> use_count_;
  const vector<vector<Coords>> input_coords_;
  const vector<Coords> ret_coords_;
};

class PipelineParser {
 public:
  Result<unique_ptr<Pipeline>> Parse(const Value& config);

 private:
  vector<string> inputs_;
  vector<string> outputs_;
  vector<unique_ptr<Node>> nodes_;
  vector<int> input_idx_;
  vector<int> output_idx_;
  vector<vector<int>> node_input_idx_;
  vector<vector<int>> node_output_idx_;
  std::map<string, int> binding_name_to_idx_;
  std::map<int, string> binding_idx_to_name_;
};

}  // namespace mmdeploy::async

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_
