// Copyright (c) OpenMMLab. All rights reserved.

#include "graph/task.h"

#include "archive/value_archive.h"
#include "core/graph.h"
#include "core/operator.h"
#include "graph/common.h"

namespace mmdeploy::graph {

static int GetDepth(const Value& input) {
  if (input.is_array() && input.size() > 0) {
    return GetDepth(input[0]) + 1;
  }
  return input.is_array();
}

// all args are array of the same length
static size_t GetBatchSize(const Value& args) {
  size_t batch_size = 0;
  for (const auto& x : args) {
    if (x.is_array()) {
      if (!batch_size) {
        batch_size = x.size();
      } else if (batch_size != x.size()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return batch_size;
}

Task::Task(const Value& cfg) : BaseNode(cfg) {
  auto module = CreateFromRegistry<Module>(cfg, "module");
  if (!module) {
    ERROR("failed to create task: {}", cfg);
    throw_exception(eFail);
  }
  module_ = std::move(module).value();
  name_ = cfg.value("name", string{});
  is_batched_ = cfg.value("is_batched", false);
  is_thread_safe_ = cfg.value("is_thread_safe", false);
}

void Task::Build(TaskGraph& graph) {
  auto handle = graph.Add([this](Context& ctx) -> Result<void> {
    auto args = ctx.pop().array();
    auto rets = Value::Array{};
    auto batch_size = GetBatchSize(args);
    //    ERROR("name: {}, is_batched: {}, INPUT batch_size: {}", name_, is_batched_, batch_size);
    if (!is_batched_ && batch_size) {
      rets.resize(outputs_.size(), Value::kArray);
      if (!is_thread_safe_) {
        for (int i = 0; i < batch_size; ++i) {
          Value sample = Value::kArray;
          for (const auto& a : args) {
            sample.push_back(a[i]);
          }
          OUTCOME_TRY(auto ret, module_->Process(sample));
          for (int j = 0; j < ret.size(); ++j) {
            rets[j].push_back(std::move(ret[j]));
          }
        }
      } else {
        std::vector<std::function<Result<Value>()>> tasks;
        tasks.reserve(batch_size);
        OUTCOME_TRY(auto batch_args, DistribAA(args));
        for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
          tasks.emplace_back([&, sample_id]() -> Result<Value> {
            return module_->Process(batch_args[sample_id]);
          });
        }
        auto batch_rets = ctx.Execute(tasks);
        for (auto& batch_ret : batch_rets) {
          OUTCOME_TRY(auto ret, std::move(batch_ret));
          for (int j = 0; j < rets.size(); ++j) {
            rets[j].push_back(std::move(ret[j]));
          }
        }
      }
    } else {
      OUTCOME_TRY(auto&& tmp, module_->Process(args));
      rets = std::move(tmp).array();
    }
    ctx.push(std::move(rets));
    //    ERROR("name: {}, is_batched: {}, OUTPUT batch_size: {}", name_, is_batched_,
    //          GetBatchSize(rets));
    return success();
  });
  handle->set_name(name_);
}

class TaskNodeCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Task"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override {
    return std::make_unique<Task>(value);
  }
};

REGISTER_MODULE(Node, TaskNodeCreator);

}  // namespace mmdeploy::graph
