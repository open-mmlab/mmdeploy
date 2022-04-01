// Copyright (c) OpenMMLab. All rights reserved.

#include "timed_single_thread_context.h"

namespace mmdeploy {

TimedSingleThreadContext::TimedSingleThreadContext() : thread_([this] { this->Run(); }) {}

TimedSingleThreadContext::~TimedSingleThreadContext() {
  {
    std::lock_guard lock{mutex_};
    stop_ = true;
    cv_.notify_one();
  }
  thread_.join();
  assert(head_ == nullptr);
}

void TimedSingleThreadContext::Enqueue(TimedSingleThreadContext::TaskBase* task) noexcept {
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

void TimedSingleThreadContext::Run() {
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

}  // namespace mmdeploy
