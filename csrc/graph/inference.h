// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PIPELINE_INFERENCE_H_
#define MMDEPLOY_SRC_PIPELINE_INFERENCE_H_

#include "graph/pipeline.h"

namespace mmdeploy::graph {

class Inference : public Node {
 public:
  static unique_ptr<Inference> Create(const Value& param);

  void Build(TaskGraph& graph) override;

 private:
  vector<string> inputs_;
  vector<string> outputs_;
  Model model_;
  unique_ptr<Pipeline> pipeline_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_PIPELINE_INFERNECE_H_
