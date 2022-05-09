// Copyright (c) OpenMMLab. All rights reserved.

#include "apis/c/executor.h"

#include "common.h"
#include "common_internal.h"
#include "execution/when_all_value.h"
#include "executor_internal.h"
#include "handle.h"

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

mmdeploy_scheduler_t mmdeploy_executor_inline() { return CreateScheduler("Inlined"); }

mmdeploy_scheduler_t mmdeploy_executor_system_pool() {
  // create a thread pool context and hold its shared handle
  static auto scheduler = *Cast(CreateScheduler("ThreadPool"));
  // return a copy of the handle to the thread pool
  return Cast(new SchedulerType(scheduler));
}

mmdeploy_scheduler_t mmdeploy_executor_create_thread_pool(int num_threads) {
  return CreateScheduler("ThreadPool", {{"num_threads", num_threads}});
}

mmdeploy_scheduler_t mmdeploy_executor_create_single_thread() {
  return CreateScheduler("SingleThread");
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

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_invocable_t fn,
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

mmdeploy_sender_t mmdeploy_executor_let_value(mmdeploy_sender_t input, mmdeploy_kleisli_t kleisli,
                                              void* context) {
  return Guard([&] {
    return Take(LetValue(Take(input), [kleisli, context](Value& args) {
      auto out = Cast(kleisli(Cast(&args), context));
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

int mmdeploy_pipeline_create(mmdeploy_value_t config, const char* device_name, int device_id,
                             mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  try {
    auto _config = *Cast(config);
    if (exec_info) {
      auto& info = _config["context"]["executor"] = Value::kObject;
      for (auto p = exec_info; p; p = p->next) {
        info[p->task_name] = *Cast(p->scheduler);
        if (p->next == exec_info) {
          MMDEPLOY_ERROR("circle detected in exec_info list.");
          return MM_E_INVALID_ARG;
        }
      }
    }
    auto _handle = std::make_unique<AsyncHandle>(device_name, device_id, std::move(_config));
    *handle = _handle.release();
    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

// TODO: handle empty input
mmdeploy_sender_t mmdeploy_pipeline_apply_async(mm_handle_t handle, mmdeploy_sender_t input) {
  if (!handle || !input) {
    return nullptr;
  }
  try {
    auto detector = static_cast<AsyncHandle*>(handle);
    return Take(detector->Process(Take(input)));
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return nullptr;
}

void mmdeploy_pipeline_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    delete static_cast<AsyncHandle*>(handle);
  }
}

int mmdeploy_pipeline_apply(mm_handle_t handle, mmdeploy_value_t input, mmdeploy_value_t* output) {
  auto input_sender = mmdeploy_executor_just(input);
  if (!input_sender) {
    return MM_E_FAIL;
  }
  auto output_sender = mmdeploy_pipeline_apply_async(handle, input_sender);
  if (!output_sender) {
    return MM_E_FAIL;
  }
  auto _output = mmdeploy_executor_sync_wait(output_sender);
  if (!_output) {
    return MM_E_FAIL;
  }
  *output = _output;
  return MM_SUCCESS;
}
