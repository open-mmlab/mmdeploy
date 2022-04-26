// Copyright (c) OpenMMLab. All rights reserved.

#include "apis/c/executor.h"

#include "core/value.h"
#include "execution/execution.h"
#include "execution/schedulers/inlined_scheduler.h"
#include "execution/schedulers/static_thread_pool.h"
#include "execution/type_erased.h"
#include "execution/when_all_value.h"

using namespace mmdeploy;

#if 1

using _Value = std::tuple<Value>;

namespace {

inline _TypeErasedScheduler<_Value>* Cast(mmdeploy_scheduler_t s) {
  return reinterpret_cast<_TypeErasedScheduler<_Value>*>(s);
}

inline mmdeploy_scheduler_t Cast(_TypeErasedScheduler<_Value>* s) {
  return reinterpret_cast<mmdeploy_scheduler_t>(s);
}

inline _TypeErasedSender<_Value>* Cast(mmdeploy_sender_t s) {
  return reinterpret_cast<_TypeErasedSender<_Value>*>(s);
}

inline mmdeploy_sender_t Cast(_TypeErasedSender<_Value>* s) {
  return reinterpret_cast<mmdeploy_sender_t>(s);
}

inline mmdeploy_value_t Cast(Value* s) { return reinterpret_cast<mmdeploy_value_t>(s); }

inline Value* Cast(mmdeploy_value_t s) { return reinterpret_cast<Value*>(s); }

}  // namespace

using Sender = _TypeErasedSender<_Value>;

mmdeploy_scheduler_t mmdeploy_inline_scheduler() {
  static auto v = new _TypeErasedScheduler<_Value>(InlineScheduler{});
  return Cast(v);
}

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value) {
  auto j = Just(*Cast(value));
  return Cast(new Sender(std::move(j)));
}

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler) {
  auto wrapped = Then(Schedule(*Cast(scheduler)), [] { return Value(); });
  return Cast(new Sender(std::move(wrapped)));
}

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler) {
  auto output_sender = ScheduleFrom(*Cast(scheduler), std::move(*Cast(input)));
  return Cast(new Sender(std::move(output_sender)));
}

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_invocable_t fn,
                                         void* context) {
  auto sender2 = Then(std::move(*Cast(input)), [fn, context](Value u) {
    auto v = Cast(fn(Cast(&u), context));
    Value w = std::move(*v);
    delete v;
    return w;
  });
  return Cast(new Sender(std::move(sender2)));
}

mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input) {
  auto split = Split(std::move(*Cast(input)));
  return Cast(new Sender(std::move(split)));
}

mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t* inputs, int32_t n) {
  std::vector<Sender> senders;
  senders.reserve(n);
  for (int i = 0; i < n; ++i) {
    senders.emplace_back(std::move(*Cast(inputs[i])));
  }
  return Cast(new Sender(
      Then(WhenAll(std::move(senders)), [](Value::Array&& v) { return Value(std::move(v)); })));
}

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input) {
  return Cast(new Value(std::get<0>(SyncWait(std::move(*Cast(input))))));
}

#endif
