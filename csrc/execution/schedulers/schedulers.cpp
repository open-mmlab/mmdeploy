// Copyright (c) OpenMMLab. All rights reserved.

#include "execution/schedulers/inlined_scheduler.h"
#include "execution/schedulers/registry.h"
#include "execution/schedulers/single_thread_context.h"
#include "execution/schedulers/static_thread_pool.h"

namespace mmdeploy {

using Scheduler = TypeErasedScheduler<Value>;

class InlineSchedulerCreator : public Creator<Scheduler> {
 public:
  const char *GetName() const override { return "Inlined"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value &) override { return ReturnType{InlineScheduler{}}; }
};

REGISTER_MODULE(Scheduler, InlineSchedulerCreator);

namespace {

// Create type-erased scheduler by calling Context::GetScheduler and then move the context into the
// deleter of the impl ptr of the type-erased scheduler
template <class Context>
Scheduler CreateFromContext(std::unique_ptr<Context> context) {
  using SchedType = decltype(context->GetScheduler());
  using EraseType = _type_erased::TypeErasedSchedulerImpl<SchedType, Value>;
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
    return CreateFromContext(std::make_unique<_single_thread_context::SingleThreadContext>());
  }
};

REGISTER_MODULE(Scheduler, SingleThreadSchedCreator);

class StaticThreadPoolSchedCreator : public Creator<Scheduler> {
 public:
  const char *GetName() const override { return "ThreadPool"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value &cfg) override {
    auto num_threads = cfg.value("num_threads", 0);
    if (num_threads) {
      return CreateFromContext(
          std::make_unique<__static_thread_pool::StaticThreadPool>(num_threads));
    } else {
      return CreateFromContext(std::make_unique<__static_thread_pool::StaticThreadPool>());
    }
  }
};

REGISTER_MODULE(Scheduler, StaticThreadPoolSchedCreator);

namespace async {

void __link_scheduler() {}

}  // namespace async

}  // namespace mmdeploy
