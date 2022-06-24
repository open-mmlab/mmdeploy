// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_GRAPH_INFERENCE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_GRAPH_INFERENCE_H_

#include "mmdeploy/core/graph.h"

namespace mmdeploy::graph {

unique_ptr<Builder> CreateInferenceBuilder(const Value &config);

// class InferenceBuilder: public Builder {
//  public:
//   explicit InferenceBuilder(Value config);
//
//  protected:
//   Result<void> SetInputs() override;
//   Result<void> SetOutputs() override;
//   Result<unique_ptr<Node>> BuildImpl() override;
// };

}

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_INFERENCE_H_
