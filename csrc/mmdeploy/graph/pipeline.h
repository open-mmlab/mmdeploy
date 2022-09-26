// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_PIPELINE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_PIPELINE_H_

#include "mmdeploy/core/graph.h"

namespace mmdeploy::graph {

class PipelineBuilder : public Builder {
 public:
  explicit PipelineBuilder(Value config);

 protected:
  Result<unique_ptr<Node>> BuildImpl() override;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_PIPELINE_H_
