// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTENED_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTENED_H_

#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/operator.h"
#include "mmdeploy/execution/expand.h"

namespace mmdeploy::graph {

class Flattened : public Node {
 public:
  Flattened(unique_ptr<Node> child, vector<bool> flatten, vector<bool> broadcast,
            vector<bool> unflatten);

  Sender<Value> Process(Sender<Value> input) override;

 private:
  const vector<bool> flatten_;
  const vector<bool> broadcast_;
  const vector<bool> unflatten_;
  unique_ptr<Node> body_;
};

inline Flattened::Flattened(unique_ptr<Node> child, vector<bool> flatten, vector<bool> broadcast,
                            vector<bool> unflatten)
    : flatten_(std::move(flatten)),
      broadcast_(std::move(broadcast)),
      unflatten_(std::move(unflatten)),
      body_(std::move(child)) {}

inline Sender<Value> Flattened::Process(Sender<Value> input) {
  auto flatten = Then([this](Value input) -> std::tuple<Value::Array, vector<int>> {
    auto [output, index] = FlattenArray(std::move(input).array(), flatten_);
    output = BroadcastArray(std::move(output), index, broadcast_);
    return {std::move(output), std::move(index)};
  });

  auto process = LetValue([this](Value::Array& v, vector<int>& idx) {
    return Just(Value(std::move(v))) | body_->Process() | Then([idx](Value output) mutable {
             return std::make_tuple(std::move(output), std::move(idx));
           });
  });

  auto unflatten = Then([this](Value output, const vector<int>& index) -> Value {
    return UnflattenArray(std::move(output).array(), index, unflatten_);
  });

  return std::move(input) | flatten | Expand() | process | Expand() | unflatten;
}

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTENED_H_
