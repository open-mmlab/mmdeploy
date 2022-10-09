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

TaskBuilder::TaskBuilder(Value config) : Builder(std::move(config)) {}

namespace {

inline Result<unique_ptr<Module>> CreateModule(const Value& config) {
  auto type = config["module"].get<std::string>();
  auto creator = Registry<Module>::Get().GetCreator(type);
  if (!creator) {
    MMDEPLOY_ERROR("failed to find module creator: {}", type);
    return Status(eEntryNotFound);
  }
  auto inst = creator->Create(config);
  if (!Check(inst)) {
    MMDEPLOY_ERROR("failed to create module: {}", type);
    return Status(eFail);
  }
  return std::move(inst);
}

}  // namespace

Result<unique_ptr<Node>> TaskBuilder::BuildImpl() {
  try {
    auto task = std::make_unique<Task>();
    OUTCOME_TRY(task->module_, CreateModule(config_));

    if (auto name = Maybe{config_} / "scheduler" / identity<string>{}) {
      if (auto sched = Maybe{config_} / "context" / "scheduler" / *name /
                       identity<TypeErasedScheduler<Value>>{}) {
        task->sched_ = std::move(*sched);
      }
    }

    if (!task->sched_) {
      task->sched_ =
          TypeErasedScheduler<Value>{std::make_shared<TypeErasedScheduler<Value>::Impl>()};
    }

    task->is_batched_ = config_.value("is_batched", false);
    task->is_thread_safe_ = config_.value("is_thread_safe", false);
    return std::move(task);
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", config_);
    return nullptr;
  }
}

class TaskCreator : public Creator<Builder> {
 public:
  const char* GetName() const override { return "Task"; }
  unique_ptr<Builder> Create(const Value& config) override {
    return std::make_unique<TaskBuilder>(config);
  }
};
REGISTER_MODULE(Builder, TaskCreator);

}  // namespace mmdeploy::graph
