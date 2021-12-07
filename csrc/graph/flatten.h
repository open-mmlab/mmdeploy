// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_GRAPH_FLATTEN_H_
#define MMDEPLOY_SRC_GRAPH_FLATTEN_H_

#include "graph/common.h"

namespace mmdeploy::graph {

class FlattenNode : public BaseNode {
 public:
  explicit FlattenNode(const Value& cfg) : BaseNode(cfg) {}
  void Build(TaskGraph& graph) override;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_GRAPH_FLATTEN_H_
