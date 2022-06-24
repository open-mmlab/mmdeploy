// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTENED_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTENED_H_

#include "mmdeploy/core/graph.h"

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

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTENED_H_
