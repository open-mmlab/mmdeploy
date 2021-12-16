// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PIPELINE_PIPELINE_H_
#define MMDEPLOY_SRC_PIPELINE_PIPELINE_H_

#include "graph/common.h"

namespace mmdeploy::graph {

class Pipeline : public BaseNode {
 public:
  explicit Pipeline(const Value& cfg);

  void Build(TaskGraph& graph) override;

 private:
  enum BindingType { kRead, kWrite };

  std::vector<int> UpdateBindings(const std::vector<std::string>& names, BindingType type);

  Result<void> Call(Context& ctx, int idx);
  Result<void> Ret(Context& ctx, int idx);

 private:
  vector<unique_ptr<Node>> nodes_;
  vector<int> input_idx_;
  vector<int> output_idx_;
  vector<vector<int>> node_input_idx_;
  vector<vector<int>> node_output_idx_;
  std::map<std::string, int> binding_name_to_idx_;
  std::map<int, std::string> binding_idx_to_name_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_PIPELINE_PIPELINE_H_
