// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_

#include <map>

#include "core/module.h"
#include "core/operator.h"
#include "core/value.h"
#include "experimental/execution/type_erased.h"
#include "experimental/execution/when_all_value.h"

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
    return Then(std::move(input), [&](const Value& v) {
      auto value = module_->Process(v).value();
      return value;
    });
  }

 private:
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
