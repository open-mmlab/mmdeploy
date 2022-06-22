// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_PIPELINE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_PIPELINE_H_

#include "mmdeploy/core/graph.h"

namespace mmdeploy::graph {

class Pipeline : public Node {
  friend class PipelineBuilder;

 public:
  Sender<Value> Process(Sender<Value> input) override;

 private:
  unique_ptr<Node> child_;
};

class PipelineBuilder : public Builder {
 public:
  explicit PipelineBuilder(Value config);
  Result<void> SetInputs() override;
  Result<void> SetOutputs() override;
  Result<unique_ptr<Node>> Build() override;

 protected:
  vector<bool> flatten_;
  vector<bool> broadcast_;
  vector<bool> unflatten_;
  vector<string> inputs_internal_;
  vector<string> outputs_internal_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_PIPELINE_H_
