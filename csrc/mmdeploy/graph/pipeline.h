// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_

#include <map>

#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/operator.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/execution/schedulers/registry.h"
#include "mmdeploy/execution/when_all_value.h"

namespace mmdeploy::graph {

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

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_PIPELINE2_H_
