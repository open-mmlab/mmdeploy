// Copyright (c) OpenMMLab. All rights reserved.

#include "apis/c/executor.h"

#include "common.h"
#include "common_internal.h"
#include "execution/when_all_value.h"
#include "executor_internal.h"

using namespace mmdeploy;

namespace {

mmdeploy_scheduler_t CreateScheduler(const char* type, const Value& config = Value()) {
  try {
    auto creator = Registry<SchedulerType>::Get().GetCreator(type);
    if (!creator) {
      MMDEPLOY_ERROR("creator for {} not found.", type);
      return nullptr;
    }
    return Cast(new SchedulerType(creator->Create(config)));
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("failed to create {}, error: {}", type, e.what());
    return nullptr;
  }
}

}  // namespace

mmdeploy_sender_t mmdeploy_sender_copy(mmdeploy_sender_t input) {
  return Take(SenderType(*Cast(input)));
}

int mmdeploy_sender_destroy(mmdeploy_sender_t sender) {
  delete Cast(sender);
  return 0;
}

mmdeploy_scheduler_t mmdeploy_executor_inline() { return CreateScheduler("Inline"); }

mmdeploy_scheduler_t mmdeploy_executor_system_pool() {
  // create a thread pool context and hold its shared handle
  static auto scheduler = *Cast(CreateScheduler("ThreadPool"));
  // return a copy of the handle to the thread pool
  return Cast(new SchedulerType(scheduler));
}

mmdeploy_scheduler_t mmdeploy_executor_create_thread_pool(int num_threads) {
  return CreateScheduler("ThreadPool", {{"num_threads", num_threads}});
}

mmdeploy_scheduler_t mmdeploy_executor_create_thread() { return CreateScheduler("SingleThread"); }

mmdeploy_scheduler_t mmdeploy_executor_dynamic_batch(mmdeploy_scheduler_t scheduler,
                                                     int max_batch_size, int timeout) {
  return CreateScheduler(
      "DynamicBatch",
      {{"scheduler", *Cast(scheduler)}, {"max_batch_size", max_batch_size}, {"timeout", timeout}});
}

int mmdeploy_scheduler_destroy(mmdeploy_scheduler_t scheduler) {
  delete Cast(scheduler);
  return 0;
}

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value) {
  return Guard([&] { return Take(Just(Take(value))); });
}

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler) {
  return Guard([&] { return Take(Then(Schedule(*Cast(scheduler)), [] { return Value(); })); });
}

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler) {
  return Guard([&] { return Take(Transfer(Take(input), *Cast(scheduler))); });
}

mmdeploy_sender_t mmdeploy_executor_on(mmdeploy_scheduler_t scheduler, mmdeploy_sender_t input) {
  return Guard([&] { return Take(On(*Cast(scheduler), Take(input))); });
}

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_then_fn_t fn,
                                         void* context) {
  return Guard([&] {
    return Take(Then(Take(input), [fn, context](Value args) {
      auto out = Cast(fn(Take(std::move(args)), context));
      Value ret(std::move(*out));
      delete out;
      return ret;
    }));
  });
}

mmdeploy_sender_t mmdeploy_executor_let_value(mmdeploy_sender_t input, mmdeploy_let_value_fn_t fn,
                                              void* context) {
  return Guard([&] {
    return Take(LetValue(Take(input), [fn, context](Value& args) {
      auto out = Cast(fn(Cast(&args), context));
      SenderType ret(std::move(*out));
      delete out;
      return ret;
    }));
  });
}

mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input) {
  return Guard([&] { return Take(Split(Take(input))); });
}

mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t* inputs, int32_t n) {
  return Guard([&] {
    std::vector<SenderType> senders;
    senders.reserve(n);
    for (int i = 0; i < n; ++i) {
      senders.emplace_back(Take(inputs[i]));
    }
    return Take(
        Then(WhenAll(std::move(senders)), [](Value::Array&& v) { return Value(std::move(v)); }));
  });
}

mmdeploy_sender_t mmdeploy_executor_ensure_started(mmdeploy_sender_t input) {
  return Guard([&] { return Take(EnsureStarted(Take(input))); });
}

int mmdeploy_executor_start_detached(mmdeploy_sender_t input) {
  try {
    StartDetached(Take(input));
    return 0;
  } catch (...) {
  }
  return -1;
}

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input) {
  return Guard([&] { return Take(std::get<Value>(SyncWait(Take(input)))); });
}
