// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/facebookexperimental/libunifex/blob/main/include/unifex/timed_single_thread_context.hpp

#pragma once

#include "mmdeploy/execution/execution.h"

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

  TimedSingleThreadContext* context_;

  template <typename Rep, typename Ratio>
  friend auto ScheduleAfter(const Scheduler& self, std::chrono::duration<Rep, Ratio> delay) noexcept
      -> __schedule_after::sender_t<std::chrono::duration<Rep, Ratio>> {
    return {self.context_, delay};
  }

  template <typename Duration = std::chrono::microseconds>
  friend __schedule_after::sender_t<Duration> tag_invoke(schedule_t,
                                                         const Scheduler& self) noexcept {
    return {self.context_, Duration::zero()};
  }
};

}  // namespace _timed_single_thread_context

class MMDEPLOY_API TimedSingleThreadContext {
  using Clock = _timed_single_thread_context::Clock;
  using Scheduler = _timed_single_thread_context::Scheduler;
  using TaskBase = _timed_single_thread_context::TaskBase;
  template <typename Duration, typename Receiver>
  friend struct _timed_single_thread_context::__schedule_after::_Operation;
  friend Scheduler;

  void Enqueue(TaskBase* task) noexcept {
    bool need_notify = false;
    {
      std::lock_guard lock{mutex_};

      if (head_ == nullptr || task->due_time_ < head_->due_time_) {
        task->next_ = head_;
        head_ = task;
        need_notify = true;
      } else {
        auto* queued_task = head_;
        // find insert pos
        while (queued_task->next_ != nullptr && queued_task->next_->due_time_ <= task->due_time_) {
          queued_task = queued_task->next_;
        }

        task->next_ = queued_task->next_;
        queued_task->next_ = task;
      }
    }
    if (need_notify) {
      cv_.notify_one();
    }
  }

  void Run() {
    std::unique_lock lock{mutex_};

    while (!stop_) {
      if (head_ != nullptr) {
        auto now = Clock::now();
        auto next_due_time = head_->due_time_;
        if (next_due_time <= now) {
          // dequeue
          auto* task = head_;
          head_ = task->next_;
          // execute
          lock.unlock();
          task->Execute();
          lock.lock();
        } else {
          cv_.wait_until(lock, next_due_time);
        }
      } else {
        cv_.wait(lock);
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;

  TaskBase* head_{nullptr};
  bool stop_{false};

  std::thread thread_;

 public:
  TimedSingleThreadContext() : thread_([this] { this->Run(); }) {}
  ~TimedSingleThreadContext() {
    {
      std::lock_guard lock{mutex_};
      stop_ = true;
      cv_.notify_one();
    }
    thread_.join();
    assert(head_ == nullptr);
  }

  Scheduler GetScheduler() noexcept { return Scheduler{*this}; }

  std::thread::id GetThreadId() const noexcept { return thread_.get_id(); }
};

namespace _timed_single_thread_context::__schedule_after {

template <typename Duration, typename Receiver>
struct _Operation<Duration, Receiver>::type : TaskBase {
  Duration duration_;
  Receiver receiver_;

  template <typename Receiver2>
  type(TimedSingleThreadContext& context, Duration duration, Receiver2&& receiver)
      : TaskBase(context, &type::ExecuteImpl),
        duration_(duration),
        receiver_((Receiver2 &&) receiver) {}

  static void ExecuteImpl(TaskBase* p) noexcept {
    auto& self = *static_cast<type*>(p);
    SetValue((Receiver &&) self.receiver_);
  }

  void Enqueue() { context_->Enqueue(this); }

  friend void tag_invoke(start_t, type& op_state) noexcept {
    op_state.due_time_ = Clock::now() + op_state.duration_;
    op_state.Enqueue();
  }
};

template <typename Duration>
struct _Sender<Duration>::type {
  using value_types = std::tuple<>;

  TimedSingleThreadContext* context_;
  Duration duration_;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend operation_t<Duration, Receiver> tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
    return {*self.context_, self.duration_, (Receiver &&) receiver};
  }
};

}  // namespace _timed_single_thread_context::__schedule_after

}  // namespace mmdeploy
