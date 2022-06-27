// Copyright (c) OpenMMLab. All rights reserved.

#include "flattened.h"

#include "mmdeploy/core/operator.h"
#include "mmdeploy/execution/expand.h"

namespace mmdeploy::graph {

Flattened::Flattened(unique_ptr<Node> child, vector<bool> flatten, vector<bool> broadcast,
                     vector<bool> unflatten)
    : flatten_(std::move(flatten)),
      broadcast_(std::move(broadcast)),
      unflatten_(std::move(unflatten)),
      body_(std::move(child)) {
  MMDEPLOY_ERROR("flatten: {}", flatten_);
  MMDEPLOY_ERROR("broadcast: {}", broadcast_);
  MMDEPLOY_ERROR("unflatten_: {}", unflatten_);
}

Sender<Value> Flattened::Process(Sender<Value> input) {
  auto flatten = Then([this](Value input) -> std::tuple<Value::Array, vector<int>> {
    MMDEPLOY_ERROR("{}", input);
    auto [output, index] = FlattenArray(std::move(input).array(), flatten_);
    MMDEPLOY_ERROR("HERE");
    output = BroadcastArray(std::move(output), index, broadcast_);
    MMDEPLOY_ERROR("HERE");
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
