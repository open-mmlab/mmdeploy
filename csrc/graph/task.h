// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PIPELINE_TASK_H_
#define MMDEPLOY_SRC_PIPELINE_TASK_H_

#include "core/graph.h"

namespace mmdeploy::graph {

class Task : public Node {
 public:
  static std::unique_ptr<Task> Create(const Value& config);
  void Build(TaskGraph& graph) override;

 protected:
  std::string name_;
  bool is_batched_{false};
  bool is_thread_safe_{false};
  std::unique_ptr<Module> module_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_PIPELINE_TASK_H_
