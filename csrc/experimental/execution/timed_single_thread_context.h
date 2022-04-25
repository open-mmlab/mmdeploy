// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "execution.h"

namespace mmdeploy {

class TimedSingleThreadContext;

namespace _timed_single_thread_context {

using Clock = std::chrono::steady_clock;
using TimePoint = typename Clock::time_point;

struct TaskBase {
  using ExecuteFn = void(TaskBase*) noexcept;

  explicit TaskBase(TimedSingleThreadContext& context, ExecuteFn* execute) noexcept
      : context_(&context), execute_(execute) {}

  TimedSingleThreadContext* const context_;
  TaskBase* next_{nullptr};
  TaskBase** prev_next_ptr_{nullptr};
  ExecuteFn* execute_;
  TimePoint due_time_;

  void Execute() noexcept { execute_(this); }
};

class Scheduler;

namespace __schedule_after {

template <typename Duration, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Duration, typename Receiver>
using operation_t = typename _Operation<Duration, remove_cvref_t<Receiver>>::type;

template <typename Duration>
struct _Sender {
  struct type;
};
template <typename Duration>
using sender_t = typename _Sender<Duration>::type;

}  // namespace __schedule_after

class Scheduler {
  friend TimedSingleThreadContext;

  explicit Scheduler(TimedSingleThreadContext& context) noexcept : context_(&context) {}

  friend bool operator==(Scheduler a, Scheduler b) noexcept { return a.context_ == b.context_; }

  friend bool operator!=(Scheduler a, Scheduler b) noexcept { return a.context_ != b.context_; }

  friend void* GetSchedulerId(const Scheduler& self) { return self.context_; }

  TimedSingleThreadContext* context_;

  template <class Rep, class Ratio>
  friend auto ScheduleAfter(const Scheduler& self, std::chrono::duration<Rep, Ratio> delay) noexcept
      -> __schedule_after::sender_t<std::chrono::duration<Rep, Ratio>> {
    return {self.context_, delay};
  }

  //  template <class Duration = std::chrono::microseconds>
  //  friend auto mmdeploySchedule(const Scheduler& self) noexcept
  //      -> __schedule_after::sender_t<Duration> {
  //    return {self.context_, Duration::zero()};
  //  }
 public:
  template <class Duration = std::chrono::microseconds>
  __schedule_after::sender_t<Duration> Schedule(const Scheduler& self) noexcept {
    return {self.context_, Duration::zero()};
  }
};

}  // namespace _timed_single_thread_context

class TimedSingleThreadContext {
  using Clock = _timed_single_thread_context::Clock;
  using Scheduler = _timed_single_thread_context::Scheduler;
  using TaskBase = _timed_single_thread_context::TaskBase;
  template <typename Duration, typename Receiver>
  friend struct _timed_single_thread_context::__schedule_after::_Operation;
  friend Scheduler;

  void Enqueue(TaskBase* task) noexcept;
  void Run();

  std::mutex mutex_;
  std::condition_variable cv_;

  TaskBase* head_{nullptr};
  bool stop_{false};

  std::thread thread_;

 public:
  TimedSingleThreadContext();
  ~TimedSingleThreadContext();

  Scheduler GetScheduler() noexcept { return Scheduler{*this}; }

  std::thread::id GetThreadId() const noexcept { return thread_.get_id(); }
};

namespace _timed_single_thread_context::__schedule_after {

template <typename Duration, typename Receiver>
struct _Operation<Duration, Receiver>::type : TaskBase {
  Duration duration_;
  Receiver receiver_;

  template <class Receiver2>
  type(TimedSingleThreadContext& context, Duration duration, Receiver2&& receiver)
      : TaskBase(context, &type::ExecuteImpl),
        duration_(duration),
        receiver_((Receiver2 &&) receiver) {}

  static void ExecuteImpl(TaskBase* p) noexcept {
    auto& self = *static_cast<type*>(p);
    SetValue((Receiver &&) self.receiver_);
  }

  void Enqueue() { context_->Enqueue(this); }

  friend void Start(type& op_state) noexcept {
    op_state.due_time_ = Clock::now() + op_state.duration_;
    op_state.Enqueue();
  }
};

template <class Duration>
struct _Sender<Duration>::type {
  using value_types = std::tuple<>;

  TimedSingleThreadContext* context_;
  Duration duration_;

  template <class Self, class Receiver, _decays_to<Self, type, int> = 0>
  friend operation_t<Duration, Receiver> Connect(Self&& self, Receiver&& receiver) {
    return {*self.context_, self.duration_, (Receiver &&) receiver};
  }
};

}  // namespace _timed_single_thread_context::__schedule_after

}  // namespace mmdeploy
