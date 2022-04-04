// Copyright (c) OpenMMLab. All rights reserved.

#include "execution.h"

#include "core/value.h"
#include "static_thread_pool.h"
#include "type_erased.h"

using namespace mmdeploy;

#if 1

using _Value = std::tuple<Value>;

namespace {

inline _TypeErasedScheduler* Cast(mmdeploy_scheduler_t s) {
  return reinterpret_cast<_TypeErasedScheduler*>(s);
}

inline mmdeploy_scheduler_t Cast(_TypeErasedScheduler* s) {
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

mmdeploy_scheduler_t mmdeploy_inline_scheduler() {
  static auto v = new _TypeErasedScheduler(InlineScheduler{});
  return Cast(v);
}

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value) {
  auto j = Just(*Cast(value));
  return Cast(new _TypeErasedSender<_Value>(std::move(j)));
}

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler) {
  auto wrapped = Then(Schedule(*Cast(scheduler)), [] { return Value(); });
  return Cast(new _TypeErasedSender<_Value>(std::move(wrapped)));
}

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler) {
  auto output_sender = ScheduleFrom(*Cast(scheduler), std::move(*Cast(input)));
  return Cast(new _TypeErasedSender<_Value>(std::move(output_sender)));
}

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_invocable_t fn,
                                         void* context) {
  auto sender2 = Then(std::move(*Cast(input)), [fn, context](Value u) {
    auto v = Cast(fn(Cast(&u), context));
    Value w = std::move(*v);
    delete v;
    return w;
  });
  return Cast(new _TypeErasedSender<_Value>(std::move(sender2)));
}

mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input) {
  auto split = Split(std::move(*Cast(input)));
  return Cast(new _TypeErasedSender<_Value>(std::move(split)));
}

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input) {
  return Cast(new Value(std::get<0>(SyncWait(std::move(*Cast(input))))));
}

#endif