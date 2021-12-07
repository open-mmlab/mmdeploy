// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_GRAPH_UNFLATTEN_H_
#define MMDEPLOY_SRC_GRAPH_UNFLATTEN_H_

#include "graph/common.h"

namespace mmdeploy::graph {

class UnflattenNode : public BaseNode {
 public:
  explicit UnflattenNode(const Value& cfg) : BaseNode(cfg) {}
  void Build(TaskGraph& graph) override;
};

}  // namespace mmdeploy::graph
#endif  // MMDEPLOY_SRC_GRAPH_UNFLATTEN_H_
