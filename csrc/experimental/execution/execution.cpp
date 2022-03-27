//
// Created by li on 2022/3/11.
//
#include "execution.h"
#include "static_thread_pool.h"
#include "type_erased.h"

using namespace mmdeploy;

#if 1

mmdeploy_scheduler_t mmdeploy_inline_scheduler() {
  static auto v = MakeTypeErasedScheduler(InlineScheduler{});
  return reinterpret_cast<mmdeploy_scheduler_t>(v);
}

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value) {
  auto sndr = Just(*reinterpret_cast<Value*>(value));
  return reinterpret_cast<mmdeploy_sender_t>(MakeTypeErasedSender(std::move(sndr)));
}

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler) {
  auto sched = reinterpret_cast<AbstractScheduler*>(scheduler);
  return reinterpret_cast<mmdeploy_sender_t>(Schedule(sched));
}

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler) {
  auto input_sndr = reinterpret_cast<AbstractSender*>(input);
  auto sched = reinterpret_cast<AbstractScheduler*>(scheduler);
  auto output_sndr = ScheduleFrom(sched, input_sndr);
  auto output = MakeTypeErasedSender(std::move(output_sndr));
  return reinterpret_cast<mmdeploy_sender_t>(output);
}

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_invocable_t fn,
                                         void* data) {
  auto sndr1 = reinterpret_cast<AbstractSender*>(input);
  auto sndr2 = Then(sndr1, [fn, data](Value u) {
    auto v = reinterpret_cast<Value*>(fn(reinterpret_cast<mmdeploy_value_t>(&u), data));
    Value w = std::move(*v);
    delete v;
    return w;
  });
  auto output = MakeTypeErasedSender(std::move(sndr2));
  return reinterpret_cast<mmdeploy_sender_t>(output);
}

mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input) {
  auto input_sender = reinterpret_cast<AbstractSender*>(input);
  auto split = Split(input_sender);
  auto output_sender = MakeTypeErasedSender(std::move(split));
  return reinterpret_cast<mmdeploy_sender_t>(output_sender);
}

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input) {
  auto _input = reinterpret_cast<AbstractSender*>(input);
  auto output = new Value(std::get<Value>(SyncWait(_input)));
  return reinterpret_cast<mmdeploy_value_t>(output);
}


#endif