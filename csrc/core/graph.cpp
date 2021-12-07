// Copyright (c) OpenMMLab. All rights reserved.

#include "core/graph.h"

#include "archive/value_archive.h"

namespace mmdeploy::graph {

TaskGraph::Handle* TaskGraph::Add(TaskFunction fn) {
  function_.push_back(std::move(fn));
  handle_.push_back(std::make_unique<Handle>());
  return handle_.back().get();
}

TaskGraph::~TaskGraph() {
  for (int i = 0; i < time_.size(); ++i) {
    INFO("node {} ({}): {} ms", i, handle_[i]->name(), static_cast<float>(time_[i]) / count_);
  }
}

Result<Value> TaskGraph::Run(Value inputs) {
  Context ctx(this);
  ctx.push(std::move(inputs));
  time_.resize(function_.size());
  for (int i = 0; i < function_.size(); ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    OUTCOME_TRY(function_[i](ctx));
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    time_[i] += dt;
  }
  count_ += 1;
  return ctx.pop();
}

std::vector<Result<Value>> TaskGraph::Execute(Span<std::function<Result<Value>()>> tasks) {
#if MMDEPLOY_USE_TASKFLOW
  std::vector<tf::Future<std::optional<Result<Value>>>> futures;
  futures.reserve(tasks.size());
  for (auto& task : tasks) {
    futures.push_back(executor_.async(task));
  }
  executor_.wait_for_all();
  std::vector<Result<Value>> rets;
  rets.reserve(tasks.size());
  for (auto& future : futures) {
    Result<Value> ret = Status(eUnknown);
    try {
      ret = *future.get();
    } catch (...) {
      ret = Status(eFail);
    }
    rets.push_back(std::move(ret));
  }
  return rets;
#else
  std::vector<Result<Value>> rets;
  rets.reserve(tasks.size());
  for (auto& task : tasks) {
    Result<Value> ret = Status(eUnknown);
    try {
      ret = task();
    } catch (const Exception& e) {
      ret = failure(e.code());
    } catch (...) {
      ret = Status(eFail);
    }
    rets.push_back(std::move(ret));
  }
  return rets;
#endif
}

std::vector<Result<Value>> Context::Execute(Span<std::function<Result<Value>()>> tasks) {
  return graph_->Execute(tasks);
}

}  // namespace mmdeploy::graph
