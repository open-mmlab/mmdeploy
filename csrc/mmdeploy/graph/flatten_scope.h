// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTEN_SCOPE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTEN_SCOPE_H_

#include "mmdeploy/core/graph.h"

namespace mmdeploy::graph {

class FlattenedScope : public Node {
 public:
  FlattenedScope(unique_ptr<Node> child, vector<bool> flatten, vector<bool> broadcast,
                 vector<bool> unflatten);

  Sender<Value> Process(Sender<Value> input) override;

 private:
  const std::vector<bool> flatten_;
  const std::vector<bool> broadcast_;
  const std::vector<bool> unflatten_;
  std::unique_ptr<Node> body_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_FLATTEN_SCOPE_H_
