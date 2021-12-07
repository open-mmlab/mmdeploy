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

unique_ptr<Task> Task::Create(const Value& config) {
  try {
    auto inst = std::make_unique<Task>();
    auto module = CreateFromRegistry<Module>(config, "module");
    if (!module) {
      ERROR("failed to create task: {}", config);
      return nullptr;
    }
    inst->module_ = std::move(module).value();
    inst->name_ = config.value("name", string{});
    inst->is_batched_ = config.value("is_batched", false);
    inst->is_thread_safe_ = config.value("is_thread_safe", false);
    from_value(config["input"], inst->inputs_);
    from_value(config["output"], inst->outputs_);
    return inst;
  } catch (...) {
    return nullptr;
  }
}
void Task::Build(TaskGraph& graph) {
  auto handle = graph.Add([this](Context& ctx) -> Result<void> {
    OUTCOME_TRY(auto args, Keys2Idxs(ctx.current(), inputs_));
    Value rets = Value::kArray;
    auto batch_size = GetBatchSize(args);
    //    ERROR("name: {}, is_batched: {}, INPUT batch_size: {}", name_, is_batched_, batch_size);
    if (!is_batched_ && batch_size) {
      for (int i = 0; i < outputs_.size(); ++i) {
        rets.push_back(Value::kArray);
      }
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
      OUTCOME_TRY(rets, module_->Process(args));
    }

    //    ERROR("name: {}, is_batched: {}, OUTPUT batch_size: {}", name_, is_batched_,
    //          GetBatchSize(rets));

    OUTCOME_TRY(Idxs2Keys(std::move(rets), outputs_, ctx.current()));
    return success();
  });
  handle->set_name(name_);
}

class TaskNodeCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Task"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& value) override { return Task::Create(value); }
};

REGISTER_MODULE(Node, TaskNodeCreator);

}  // namespace mmdeploy::graph
