// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_

#include "pipeline.h"

namespace mmdeploy::graph {

class Inference : public Node {
  friend class InferenceParser;

 public:
  Sender<Value> Process(Sender<Value> input) override {
    return pipeline_->Process(std::move(input));
  }

  unique_ptr<Pipeline> pipeline_;
};

class InferenceParser {
 public:
  static Result<unique_ptr<Inference>> Parse(const Value& config);
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_
