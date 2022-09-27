// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_RUN_LOOP_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_RUN_LOOP_H_

#include <condition_variable>
#include <mutex>

#include "concepts.h"
#include "utility.h"

namespace mmdeploy {

namespace __loop {
class RunLoop;

namespace __impl {

struct _Task {
  virtual void _Execute() noexcept = 0;
  _Task* next_ = nullptr;
};

template <typename Receiver>
struct _Operation {
  struct type;
};
template <typename Receiver>
using operation_t = typename _Operation<remove_cvref_t<Receiver>>::type;

template <typename Receiver>
struct _Operation<Receiver>::type final : _Task {
  friend void tag_invoke(start_t, type& op_state) noexcept { op_state._Start(); }

  void _Execute() noexcept override { SetValue(std::move(receiver_)); }
  void _Start() noexcept;

  Receiver receiver_;
  RunLoop* const loop_;

 public:
  template <class _Receiver2>
  explicit type(_Receiver2&& receiver, RunLoop* loop)
      : receiver_((_Receiver2 &&) receiver), loop_(loop) {}
};

}  // namespace __impl

class RunLoop {
  template <typename>
  friend struct __impl::_Operation;

 public:
  class _Scheduler {
    struct _ScheduleTask {
      using value_types = std::tuple<>;

     private:
      friend _Scheduler;

      template <typename Receiver>
      friend __impl::operation_t<Receiver> tag_invoke(connect_t, const _ScheduleTask& self,
                                                      Receiver&& receiver) {
        return __impl::operation_t<Receiver>{(Receiver &&) receiver, self.loop_};
      }
      RunLoop* const loop_;

     public:
      explicit _ScheduleTask(RunLoop* loop) noexcept : loop_(loop) {}

      friend _Scheduler tag_invoke(get_completion_scheduler_t, const _ScheduleTask& self);
    };
    friend RunLoop;

    friend _Scheduler tag_invoke(get_completion_scheduler_t, const _ScheduleTask& self) {
      return RunLoop::_Scheduler{self.loop_};
    }

    explicit _Scheduler(RunLoop* loop) noexcept : loop_(loop) {}

   public:
    bool operator==(const _Scheduler& other) const noexcept { return loop_ == other.loop_; }

    _Scheduler(const _Scheduler& other) = default;

   private:
    friend _ScheduleTask tag_invoke(schedule_t, const _Scheduler& self) {
      return _ScheduleTask{self.loop_};
    }
    RunLoop* loop_;
  };
  _Scheduler GetScheduler() { return _Scheduler{this}; }
  void _Run();
  void _Finish();

 private:
  void _push_back(__impl::_Task* task);
  __impl::_Task* _pop_front();

  std::mutex mutex_;
  std::condition_variable cv_;
  __impl::_Task* head_ = nullptr;
  __impl::_Task* tail_ = nullptr;
  bool stop_ = false;
};

namespace __impl {

template <typename Receiver>
inline void _Operation<Receiver>::type::_Start() noexcept {
  loop_->_push_back(this);
}

}  // namespace __impl

inline void RunLoop::_Run() {
  while (auto* task = _pop_front()) {
    task->_Execute();
  }
}

inline void RunLoop::_Finish() {
  std::lock_guard lock{mutex_};
  stop_ = true;
  cv_.notify_all();
}

inline void RunLoop::_push_back(__impl::_Task* task) {
  std::lock_guard lock{mutex_};
  if (head_ == nullptr) {
    head_ = task;
  } else {
    tail_->next_ = task;
  }
  tail_ = task;
  task->next_ = nullptr;
  cv_.notify_one();
}

inline __impl::_Task* RunLoop::_pop_front() {
  std::unique_lock lock{mutex_};
  while (head_ == nullptr) {
    if (stop_) {
      return nullptr;
    }
    cv_.wait(lock);
  }
  auto* task = head_;
  head_ = task->next_;
  if (head_ == nullptr) {
    tail_ = nullptr;
  }
  return task;
}

}  // namespace __loop

using RunLoop = __loop::RunLoop;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_RUN_LOOP_H_
