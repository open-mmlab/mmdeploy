//
// Created by li on 2022/3/11.
//
#include "execution.h"

#include "core/value.h"
#include "static_thread_pool.h"
#include "type_erased.h"

using namespace mmdeploy;

#if 1

using _Value = std::tuple<Value>;

mmdeploy_scheduler_t mmdeploy_inline_scheduler() {
  static auto v = new _TypeErasedScheduler(InlineScheduler{});
  return reinterpret_cast<mmdeploy_scheduler_t>(v);
}

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value) {
  auto j = Just(*reinterpret_cast<Value*>(value));
  auto s = new _TypeErasedSender<_Value>(std::move(j));
  return reinterpret_cast<mmdeploy_sender_t>(s);
}

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler) {
  auto sched = reinterpret_cast<_TypeErasedScheduler*>(scheduler);
  auto wrapped = Then(Schedule(*sched), [] { return Value(); });
  return reinterpret_cast<mmdeploy_sender_t>(new _TypeErasedSender<_Value>(std::move(wrapped)));
}

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler) {
  auto input_sender = reinterpret_cast<_TypeErasedSender<_Value>*>(input);
  auto sched = reinterpret_cast<_TypeErasedScheduler*>(scheduler);
  auto output_sender = ScheduleFrom(*sched, std::move(*input_sender));
  auto output = new _TypeErasedSender<_Value>(std::move(output_sender));
  return reinterpret_cast<mmdeploy_sender_t>(output);
}

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_invocable_t fn,
                                         void* context) {
  auto sender1 = reinterpret_cast<_TypeErasedSender<_Value>*>(input);
  auto sender2 = Then(std::move(*sender1), [fn, context](Value u) {
    auto v = reinterpret_cast<Value*>(fn(reinterpret_cast<mmdeploy_value_t>(&u), context));
    Value w = std::move(*v);
    delete v;
    return w;
  });
  auto output = new _TypeErasedSender<_Value>(std::move(sender2));
  return reinterpret_cast<mmdeploy_sender_t>(output);
}

mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input) {
  auto input_sender = reinterpret_cast<_TypeErasedSender<_Value>*>(input);
  auto split = Split(std::move(*input_sender));
  auto output_sender = new _TypeErasedSender<_Value>(std::move(split));
  return reinterpret_cast<mmdeploy_sender_t>(output_sender);
}

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input) {
  auto _input = reinterpret_cast<_TypeErasedSender<_Value>*>(input);
  auto output = new Value(std::get<0>(SyncWait(std::move(*_input))));
  return reinterpret_cast<mmdeploy_value_t>(output);
}

#endif