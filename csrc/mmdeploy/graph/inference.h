// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_INFERENCE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_INFERENCE_H_

#include "mmdeploy/core/graph.h"

namespace mmdeploy::graph {

class InferenceBuilder : public Builder {
 public:
  explicit InferenceBuilder(Value config);

 protected:
  Result<unique_ptr<Node>> BuildImpl() override;

 private:
  Result<void> CheckInputs(Builder& builder);
  Result<void> CheckOutputs(Builder& builder);
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_INFERENCE_H_
