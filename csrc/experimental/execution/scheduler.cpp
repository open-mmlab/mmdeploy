// Copyright (c) OpenMMLab. All rights reserved.

#include "inlined_scheduler.h"
#include "pipeline2.h"
#include "static_thread_pool.h"
#include "timed_single_thread_context.h"

namespace mmdeploy {

using Scheduler = TypeErasedScheduler<Value>;

class InlineSchedulerCreator : public Creator<Scheduler> {
 public:
  const char *GetName() const override { return "Inline"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value &) override { return ReturnType{InlineScheduler{}}; }
};

REGISTER_MODULE(Scheduler, InlineSchedulerCreator);

namespace {

template <class Context>
Scheduler CreateFromContext(std::unique_ptr<Context> context) {
  using SchedType = decltype(context->GetScheduler());
  using EraseType = TypeErasedSchedulerImpl<SchedType, Value>;
  auto sched = new EraseType(context->GetScheduler());
  return Scheduler{std::shared_ptr<Scheduler::Impl>(
      sched, [context = std::move(context)](EraseType *p) { delete p; })};
}

}  // namespace

class SingleThreadSchedCreator : public Creator<Scheduler> {
 public:
  const char *GetName() const override { return "SingleThread"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value &) override {
    return CreateFromContext(std::make_unique<TimedSingleThreadContext>());
  }
};

REGISTER_MODULE(Scheduler, SingleThreadSchedCreator);

class StaticThreadPoolSchedCreator : public Creator<Scheduler> {
 public:
  const char *GetName() const override { return "ThreadPool"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value &) override {
    return CreateFromContext(std::make_unique<__static_thread_pool::StaticThreadPool>());
  }
};

REGISTER_MODULE(Scheduler, StaticThreadPoolSchedCreator);

}  // namespace mmdeploy
