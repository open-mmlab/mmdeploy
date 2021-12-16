// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PIPELINE_INFERENCE_H_
#define MMDEPLOY_SRC_PIPELINE_INFERENCE_H_

#include "graph/pipeline.h"

namespace mmdeploy::graph {

class Inference : public BaseNode {
 public:
  explicit Inference(const Value& cfg);

  void Build(TaskGraph& graph) override;

 private:
  Model model_;
  unique_ptr<Pipeline> pipeline_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_PIPELINE_INFERNECE_H_
