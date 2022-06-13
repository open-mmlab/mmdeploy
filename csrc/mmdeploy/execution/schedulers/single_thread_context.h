// Copyright (c) OpenMMLab. All rights reserved.
// Copied from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/examples/schedulers/single_thread_context.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SINGLE_THREAD_CONTEXT_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SINGLE_THREAD_CONTEXT_H_

#include <thread>

#include "mmdeploy/execution/run_loop.h"

namespace mmdeploy {

namespace _single_thread_context {

class SingleThreadContext {
 public:
  SingleThreadContext() : loop_(), thread_([this] { loop_._Run(); }) {}

  ~SingleThreadContext() {
    loop_._Finish();
    thread_.join();
  }

  class Scheduler {
   public:
    explicit Scheduler(SingleThreadContext* context)
        : context_(context), scheduler_(context_->loop_.GetScheduler()) {}

    friend auto tag_invoke(schedule_t, const Scheduler& self)
        -> tag_invoke_result_t<schedule_t, RunLoop::_Scheduler> {
      return Schedule(self.scheduler_);
    }

   private:
    SingleThreadContext* context_;
    RunLoop::_Scheduler scheduler_;
  };

  Scheduler GetScheduler() noexcept { return Scheduler{this}; }

  std::thread::id GetThreadId() const noexcept { return thread_.get_id(); }

 private:
  RunLoop loop_;
  std::thread thread_;
};

using Scheduler = SingleThreadContext::Scheduler;

}  // namespace _single_thread_context

using _single_thread_context::SingleThreadContext;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SINGLE_THREAD_CONTEXT_H_
