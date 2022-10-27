// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_COND_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_COND_H_

#include "mmdeploy/core/graph.h"

namespace mmdeploy::graph {

class Cond : public Node {
  friend class CondBuilder;

 public:
  Sender<Value> Process(Sender<Value> input) override;

 private:
  std::unique_ptr<Node> node_;
  int n_output_{0};
};

class CondBuilder : public Builder {
 public:
  explicit CondBuilder(Value config);

 protected:
  Result<unique_ptr<Node>> BuildImpl() override;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_COND_H_
