// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PIPELINE_PIPELINE_H_
#define MMDEPLOY_SRC_PIPELINE_PIPELINE_H_

#include "core/graph.h"

namespace mmdeploy::graph {

class Pipeline : public Node {
 public:
  static unique_ptr<Pipeline> Create(const Value& config);

  void Build(TaskGraph& graph) override;

 private:
  vector<unique_ptr<Node> > nodes_;
  vector<string> inputs_;
  vector<string> outputs_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_PIPELINE_PIPELINE_H_
