// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/task.h"

#include "mmdeploy/core/operator.h"
#include "mmdeploy/graph/common.h"

namespace mmdeploy::graph {

Sender<Value> Task::Process(Sender<Value> input) {
  return LetValue(std::move(input), [this](Value& v) -> Sender<Value> {
    assert(v.is_array());
    // handle empty input
    if (v.front().empty()) {
      return TransferJust(*sched_, Value(Value::Array(v.size(), Value::kArray)));
    }
    if (v.front().is_array() && !is_batched_) {
      auto batch_size = v.front().size();
      Value output = Value::Array(batch_size);
      // clang-format off
      return TransferJust(*sched_, std::move(output))
          | Then([&](Value&& output) -> Value {
            auto input = graph::DistribAA(v).value();
            return Value{std::move(input), std::move(output)};
          })
          | Bulk(batch_size, [&](size_t index, Value& in_out) {
            const auto& input = in_out[0];
            auto& output = in_out[1];
            output[index] = module_->Process(input[index]).value();
          })
          | Then([](const Value& in_out) {
            return graph::DistribAA(in_out[1]).value();
          });
      // clang-format on
    } else {
      return DynamicBatch(TransferJust(*sched_, std::move(v)), batch_context_,
                          [&](const Value& u) { return module_->Process(u).value(); });
    }
  });
}

Result<unique_ptr<Task>> TaskParser::Parse(const Value& config) {
  try {
    auto task = std::make_unique<Task>();
    OUTCOME_TRY(NodeParser::Parse(config, *task));
    OUTCOME_TRY(task->module_, CreateFromRegistry<Module>(config, "module"));
    bool sched_set = false;
    if (config["context"].contains("executor")) {
      auto& exec_info = config["context"]["executor"];
      for (auto it = exec_info.begin(); it != exec_info.end(); ++it) {
        if (it.key() == task->name()) {
          task->sched_ = it->get<TypeErasedScheduler<Value>>();
          sched_set = true;
          MMDEPLOY_INFO("scheduler configured for task {}", task->name());
          break;
        }
      }
    }
    if (!sched_set) {
      task->sched_ =
          TypeErasedScheduler<Value>{std::make_shared<TypeErasedScheduler<Value>::Impl>()};
    }
    task->is_batched_ = config.value("is_batched", false);
    task->is_thread_safe_ = config.value("is_thread_safe", false);
    return std::move(task);
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", config);
    return nullptr;
  }
}

class TaskCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Task"; }
  std::unique_ptr<Node> Create(const Value& value) override {
    try {
      return TaskParser::Parse(value).value();
    } catch (...) {
    }
    return nullptr;
  }
};

REGISTER_MODULE(Node, TaskCreator);

}  // namespace mmdeploy::graph
