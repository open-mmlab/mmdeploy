// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_

#include "core/model.h"
#include "core/module.h"
#include "core/registry.h"
#include "core/status_code.h"
#include "mpl/span.h"
#include "utils/formatter.h"

#if MMDEPLOY_USE_TASKFLOW
#include "taskflow/taskflow.hpp"
#endif

namespace mmdeploy::graph {

using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

class TaskGraph;
class Node;

class Context {
 public:
  explicit Context(TaskGraph* graph) : graph_(graph) {}

  Value& current() { return context_.back(); }

  void push(Value&& ctx) { context_.push_back(std::move(ctx)); }

  Value pop() {
    auto ctx = std::move(context_.back());
    context_.pop_back();
    return ctx;
  }

  size_t size() const noexcept { return context_.size(); }
  bool empty() const noexcept { return context_.empty(); }

  std::vector<Result<Value>> Execute(Span<std::function<Result<Value>()>> tasks);

 private:
  vector<Value> context_;
  TaskGraph* graph_;
};

class TaskGraph {
  friend class Context;

 public:
  using TaskFunction = std::function<Result<void>(Context& ctx)>;

  class Handle {
   public:
    const std::string& name() const noexcept { return name_; }
    void set_name(const std::string& name) { name_ = name; }

   private:
    std::string name_;
  };

  ~TaskGraph();

  Handle* Add(TaskFunction fn);

  Result<Value> Run(Value inputs);

 private:
  std::vector<Result<Value>> Execute(Span<std::function<Result<Value>()>> tasks);

  vector<TaskFunction> function_;
  vector<unique_ptr<Handle>> handle_;
#if MMDEPLOY_USE_TASKFLOW
  tf::Executor executor_;
#endif
  // profiling utils
  std::vector<double> time_;
  int64_t count_{};
};

class Node {
 public:
  virtual ~Node() = default;
  virtual void Build(TaskGraph& graph) = 0;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_
