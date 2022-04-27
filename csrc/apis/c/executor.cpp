// Copyright (c) OpenMMLab. All rights reserved.

#include "apis/c/executor.h"

#include "core/value.h"
#include "execution/execution.h"
#include "execution/schedulers/registry.h"
#include "execution/type_erased.h"
#include "execution/when_all_value.h"

using namespace mmdeploy;

using SenderType = TypeErasedSender<Value>;
using SchedulerType = TypeErasedScheduler<Value>;

namespace {

inline SchedulerType* Cast(mmdeploy_scheduler_t s) { return reinterpret_cast<SchedulerType*>(s); }

inline mmdeploy_scheduler_t Cast(SchedulerType* s) {
  return reinterpret_cast<mmdeploy_scheduler_t>(s);
}

inline SenderType* Cast(mmdeploy_sender_t s) { return reinterpret_cast<SenderType*>(s); }

inline mmdeploy_sender_t Cast(SenderType* s) { return reinterpret_cast<mmdeploy_sender_t>(s); }

inline mmdeploy_value_t Cast(Value* s) { return reinterpret_cast<mmdeploy_value_t>(s); }

inline Value* Cast(mmdeploy_value_t s) { return reinterpret_cast<Value*>(s); }

inline SenderType Take(mmdeploy_sender_t s) {
  auto sender = std::move(*Cast(s));
  mmdeploy_sender_destroy(s);
  return sender;
}

inline mmdeploy_sender_t Take(SenderType s) { return Cast(new SenderType(std::move(s))); }

template <typename T, std::enable_if_t<_is_sender<T>, int> = 0>
inline mmdeploy_sender_t Take(T& s) {
  return Take(SenderType(std::move(s)));
}

inline Value Take(mmdeploy_value_t v) {
  auto value = std::move(*Cast(v));
  mmdeploy_value_destroy(v);
  return value;
}

mmdeploy_value_t Take(Value v) {
  return Cast(new Value(std::move(v)));  // NOLINT
}

mmdeploy_scheduler_t CreateScheduler(const char* type) {
  try {
    auto creator = Registry<SchedulerType>::Get().GetCreator(type);
    return Cast(new SchedulerType(creator->Create(Value::kNull)));
  } catch (...) {
    return nullptr;
  }
}

}  // namespace

template <typename F>
std::invoke_result_t<F> Guard(F f) {
  try {
    return f();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return nullptr;
}

mmdeploy_sender_t mmdeploy_sender_copy(mmdeploy_sender_t input) {
  return Take(SenderType(*Cast(input)));
}

int mmdeploy_sender_destroy(mmdeploy_sender_t sender) {
  delete Cast(sender);
  return 0;
}

mmdeploy_value_t mmdeploy_value_copy(mmdeploy_value_t input) {
  return Guard([&] { return Take(Value(*Cast(input))); });
}

int mmdeploy_value_destroy(mmdeploy_value_t value) {
  delete Cast(value);
  return 0;
}

mmdeploy_scheduler_t mmdeploy_inline_scheduler() { return CreateScheduler("Inlined"); }

mmdeploy_scheduler_t mmdeploy_system_pool_scheduler() { return CreateScheduler("ThreadPool"); }

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
