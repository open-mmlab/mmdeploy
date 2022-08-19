// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/execution/schedulers/dynamic_batch_scheduler.h"
#include "mmdeploy/execution/schedulers/inlined_scheduler.h"
#include "mmdeploy/execution/schedulers/registry.h"
#include "mmdeploy/execution/schedulers/single_thread_context.h"
#include "mmdeploy/execution/schedulers/static_thread_pool.h"
#include "mmdeploy/execution/schedulers/timed_single_thread_context.h"

namespace mmdeploy {

using Scheduler = TypeErasedScheduler<Value>;

class InlineSchedulerCreator : public Creator<Scheduler> {
 public:
  const char* GetName() const override { return "Inline"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value&) override { return ReturnType{InlineScheduler{}}; }
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
      sched, [context = std::shared_ptr<Context>(std::move(context))](EraseType* p) { delete p; })};
}

}  // namespace

class SingleThreadSchedCreator : public Creator<Scheduler> {
 public:
  const char* GetName() const override { return "SingleThread"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value&) override {
    return CreateFromContext(std::make_unique<_single_thread_context::SingleThreadContext>());
  }
};

REGISTER_MODULE(Scheduler, SingleThreadSchedCreator);

class StaticThreadPoolSchedCreator : public Creator<Scheduler> {
 public:
  const char* GetName() const override { return "ThreadPool"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value& cfg) override {
    auto num_threads = -1;
    if (cfg.is_object() && cfg.contains("num_threads")) {
      num_threads = cfg["num_threads"].get<int>();
    }
    if (num_threads >= 1) {
      return CreateFromContext(
          std::make_unique<__static_thread_pool::StaticThreadPool>(num_threads));
    } else {
      return CreateFromContext(std::make_unique<__static_thread_pool::StaticThreadPool>());
    }
  }
};

REGISTER_MODULE(Scheduler, StaticThreadPoolSchedCreator);

struct ValueAssembler {
  using range_t = std::pair<size_t, size_t>;

  static size_t get_size(const Value& x) { return x.empty() ? 0 : x.front().size(); }

  template <typename ValueType>
  static void input(std::tuple<ValueType> _src, range_t src_range, std::tuple<Value>& _dst,
                    range_t dst_range, size_t batch_size) {
    auto& [src] = _src;
    auto& [dst] = _dst;
    if (dst.empty()) {
      dst = std::move(src);
      for (auto& x : dst) {
        x.array().reserve(batch_size);
      }
      return;
    }
    auto& u = src.array();
    auto& v = dst.array();
    assert(u.size() == v.size());
    assert(dst_range.first = v.front().size());
    for (size_t k = 0; k < src.size(); ++k) {
      auto& x = u[k].array();
      auto& y = v[k].array();
      std::copy(std::begin(x) + src_range.first, std::begin(x) + src_range.first + src_range.second,
                std::back_inserter(y));
    }
  }

  static void output(Value& src, range_t src_range, Value& dst, range_t dst_range,
                     size_t batch_size) {
    if (dst.empty()) {
      dst = Value::Array(src.size(), Value::Array(batch_size));
    }
    auto& u = src.array();
    auto& v = dst.array();
    assert(u.size() == v.size());
    for (size_t k = 0; k < src.size(); ++k) {
      auto& x = u[k].array();
      auto& y = v[k].array();
      std::move(std::begin(x) + src_range.first, std::begin(x) + src_range.first + src_range.second,
                std::begin(y) + dst_range.first);
    }
  }
};

TimedSingleThreadContext& gTimedSingleThreadContext() {
  static TimedSingleThreadContext context{};
  return context;
}

class DynamicBatchSchedCreator : public Creator<Scheduler> {
 public:
  const char* GetName() const override { return "DynamicBatch"; }
  int GetVersion() const override { return 0; }
  ReturnType Create(const Value& cfg) override {
    using SchedulerType =
        DynamicBatchScheduler<InlineScheduler, TypeErasedScheduler<Value>, ValueAssembler>;
    auto scheduler = cfg["scheduler"].get<TypeErasedScheduler<Value>>();
    auto max_batch_size = cfg["max_batch_size"].get<int>();

    TimedSingleThreadContext* timer{};
    auto timeout = cfg["timeout"].get<int>();
    if (timeout >= 0) {
      timer = &gTimedSingleThreadContext();
    }
    return ReturnType{SchedulerType{inline_scheduler, std::move(scheduler), timer,
                                    (size_t)max_batch_size, std::chrono::microseconds(timeout)}};
  }
};

REGISTER_MODULE(Scheduler, DynamicBatchSchedCreator);

}  // namespace mmdeploy
