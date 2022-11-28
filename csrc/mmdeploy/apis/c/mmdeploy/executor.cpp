// Copyright (c) OpenMMLab. All rights reserved.

#include "executor.h"

#include "common.h"
#include "common_internal.h"
#include "executor_internal.h"
#include "mmdeploy/execution/when_all_value.h"

using namespace mmdeploy;

namespace {

mmdeploy_scheduler_t CreateScheduler(const char* type, const Value& config = Value()) {
  try {
    auto creator = gRegistry<SchedulerType>().Get(type);
    if (!creator) {
      MMDEPLOY_ERROR("Creator for {} not found. Available schedulers: {}", type,
                     gRegistry<SchedulerType>().List());
      return nullptr;
    }
    return Cast(new SchedulerType(creator->Create(config)));
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("failed to create Scheduler: {} ({}), config: {}", type, e.what(), config);
    return nullptr;
  }
}

}  // namespace

mmdeploy_sender_t mmdeploy_sender_copy(mmdeploy_sender_t input) {
  if (!input) {
    return nullptr;
  }
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
  if (!scheduler) {
    return nullptr;
  }
  return CreateScheduler(
      "DynamicBatch",
      {{"scheduler", *Cast(scheduler)}, {"max_batch_size", max_batch_size}, {"timeout", timeout}});
}

int mmdeploy_scheduler_destroy(mmdeploy_scheduler_t scheduler) {
  delete Cast(scheduler);
  return 0;
}

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value) {
  if (value) {
    return Guard([&] { return Take(Just(*Cast(value))); });
  } else {
    return Take(Just(Value()));
  }
}

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler) {
  if (!scheduler) {
    return nullptr;
  }
  return Guard([&] { return Take(Then(Schedule(*Cast(scheduler)), [] { return Value(); })); });
}

mmdeploy_sender_t mmdeploy_executor_transfer_just(mmdeploy_scheduler_t scheduler,
                                                  mmdeploy_value_t value) {
  if (!scheduler || !value) {
    return nullptr;
  }
  return Guard([&] { return Take(TransferJust(*Cast(scheduler), *Cast(value))); });
}

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler) {
  if (!input || !scheduler) {
    return nullptr;
  }
  return Guard([&] { return Take(Transfer(Take(input), *Cast(scheduler))); });
}

mmdeploy_sender_t mmdeploy_executor_on(mmdeploy_scheduler_t scheduler, mmdeploy_sender_t input) {
  if (!scheduler || !input) {
    return nullptr;
  }
  return Guard([&] { return Take(On(*Cast(scheduler), Take(input))); });
}

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_then_fn_t fn,
                                         void* context) {
  if (!input || !fn) {
    return nullptr;
  }
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
  if (!input || !fn) {
    return nullptr;
  }
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
  if (!input) {
    return nullptr;
  }
  return Guard([&] { return Take(Split(Take(input))); });
}

mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t inputs[], int32_t n) {
  if (!inputs) {
    return nullptr;
  }
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
  if (!input) {
    return nullptr;
  }
  return Guard([&] { return Take(EnsureStarted(Take(input))); });
}

int mmdeploy_executor_start_detached(mmdeploy_sender_t input) {
  if (!input) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    StartDetached(Take(input));
    return 0;
  } catch (...) {
  }
  return MMDEPLOY_E_FAIL;
}

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input) {
  if (!input) {
    return nullptr;
  }
  return Guard([&] { return Take(std::get<Value>(SyncWait(Take(input)))); });
}

int mmdeploy_executor_sync_wait_v2(mmdeploy_sender_t sender, mmdeploy_value_t* value) {
  if (!sender) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  auto result = mmdeploy_executor_sync_wait(sender);
  if (!result) {
    return MMDEPLOY_E_FAIL;
  }
  if (value) {
    *value = result;
  } else {
    mmdeploy_value_destroy(result);
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_executor_execute(mmdeploy_scheduler_t scheduler, void (*fn)(void*), void* context) {
  Execute(*Cast(scheduler), [fn, context] { fn(context); });
}
