// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_PIPELINE_TASK_H_
#define MMDEPLOY_SRC_PIPELINE_TASK_H_

#include "graph/common.h"

namespace mmdeploy::graph {

class Task : public BaseNode {
 public:
  explicit Task(const Value& cfg);
  void Build(TaskGraph& graph) override;

 protected:
  bool is_batched_{false};
  bool is_thread_safe_{false};
  std::unique_ptr<Module> module_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_PIPELINE_TASK_H_
