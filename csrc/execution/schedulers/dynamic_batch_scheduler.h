// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_DYNAMIC_BATCH_SCHEDULER_H_
#define MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_DYNAMIC_BATCH_SCHEDULER_H_

#include "core/utils/formatter.h"
#include "execution/dynamic_batch.h"
#include "execution/schedulers/timed_single_thread_context.h"
#include "execution/utility.h"

namespace mmdeploy {

namespace _dynamic_batch_scheduler {

template <typename SubmitSch, typename ExecuteSch>
struct DynamicBatchScheduler {
  SubmitSch submit_sch_;
  ExecuteSch execute_sch_;
  TimedSingleThreadContext* timer_;
  size_t max_batch_size_;
  std::chrono::duration<int64_t, std::micro> timeout_;

  friend auto tag_invoke(schedule_t, const DynamicBatchScheduler& self) {
    return Schedule(self.submit_sch_);
  }
};

template <typename SubmitSch, typename ExecuteSch>
using scheduler_t = DynamicBatchScheduler<SubmitSch, ExecuteSch>;

template <typename Sender, typename Scheduler, typename Manager, typename Receiver, typename Func>
struct _Operation {
  struct type;
};
template <typename Sender, typename Scheduler, typename Manager, typename Receiver, typename Func>
using operation_t =
    typename _Operation<Sender, Scheduler, Manager, remove_cvref_t<Receiver>, Func>::type;

template <typename Sender, typename Scheduler, typename Manager, typename Receiver, typename Func>
struct _Receiver {
  struct type {
    operation_t<Sender, Scheduler, Manager, Receiver, Func>* op_state_;
    template <typename... Args>
    friend void tag_invoke(set_value_t, type&& self, Args&&... args) noexcept {
      self.op_state_->context_->Notify(self.op_state_, (Args &&) args...);
    }
  };
};
template <typename Sender, typename Scheduler, typename Manager, typename Receiver, typename Func>
using receiver_t = typename _Receiver<Sender, Scheduler, Manager, Receiver, Func>::type;

using context_base_t = dynamic_batch_t::context_base_t;

using range_t = std::pair<int, unsigned>;

template <typename Sender, typename Scheduler, typename Manager, typename Receiver, typename Func>
struct Context : context_base_t {
  using _duration_t = std::chrono::duration<int64_t, std::micro>;

  Scheduler scheduler_;
  Manager* manager_;
  Func func_;
  size_t max_batch_size_;
  _duration_t delay_;
  TimedSingleThreadContext* timer_;

  std::mutex mutex_;
  size_t counter_{0};

  Context(Scheduler scheduler, Manager* manager, Func func)
      : context_base_t{[](context_base_t* p) { delete p; }},
        scheduler_(std::move(scheduler)),
        manager_(manager),
        func_(std::move(func)),
        max_batch_size_(scheduler_.max_batch_size_),
        delay_(scheduler_.timeout_),
        timer_(scheduler_.timer_) {}

  using _operation_t = operation_t<Sender, Scheduler, Manager, Receiver, Func>;

  struct Batch {
    Context* context_;
    size_t index_{0};
    std::vector<_operation_t*> states_;
    std::vector<range_t> ranges_;
    completion_signatures_of_t<Sender> values_;
    size_t size_{0};
    Batch(Context* context, size_t index, size_t max_batch_size)
        : context_(context), index_(index), values_{} {
      states_.reserve(max_batch_size);
      ranges_.reserve(max_batch_size);
    }

    friend std::ostream& operator<<(std::ostream& os, const Batch& batch) {
      os << fmt::format("(index={}, size={})", batch.index_, batch.size_);
    }
  };

  template <typename... Args>
  void Notify(_operation_t* op_state, Args&&... args) {
    std::lock_guard lock{mutex_};

    std::unique_ptr<Batch> batch = std::move(batch_);
    const size_t size = Manager::get_size((Args &&) args...);
    op_state->count_ = size;

    size_t index = 0;
    while (index != size) {
      bool new_batch{};
      if (!batch) {
        batch = std::make_unique<Batch>(this, counter_++, max_batch_size_);
        new_batch = true;
      }
      auto count = std::min(max_batch_size_ - batch->size_, size - index);
      auto start = index;

      batch->states_.push_back(op_state);
      batch->ranges_.emplace_back(start, count);
      Manager::input(std::forward_as_tuple((Args &&) args...), {start, count}, batch->values_,
                     {batch->size_, count});
      batch->size_ += count;

      index += count;
      if (batch->size_ == max_batch_size_) {
        MMDEPLOY_DEBUG("direct submit of batch {}", *batch);
        // batch is full, submit immediately
        Execute(scheduler_.execute_sch_, [this, batch = std::move(batch)] { Run(*batch); });
      } else if (new_batch && timer_) {
        MMDEPLOY_DEBUG("set off deferred submission for batch {}", *batch);
        // set off a deferred task to submit the batch if it still exists at the moment.
        StartDetached(Then(ScheduleAfter(timer_->GetScheduler(), delay_),
                           [this, batch_index = batch->index_] { Submit(batch_index); }));
      }
    }

    batch_ = std::move(batch);
  }

  void Submit(int batch_index) {
    Execute(scheduler_.execute_sch_, [this, batch_index] {
      std::unique_ptr<Batch> batch;
      {
        std::lock_guard lock{mutex_};
        if (batch_ && batch_->index_ == batch_index) {
          batch = std::move(batch_);
        } else {
          MMDEPLOY_DEBUG("batch index mismatch, signal canceled ({} vs {})", batch_index,
                         (batch_ ? (int)batch_->index_ : -1));
        }
      }
      if (batch) {
        MMDEPLOY_DEBUG("deferred submit of batch {}", *batch);
        Run(*batch);
      }
    });
  }

  void Run(Batch& batch) {
    auto rets = std::apply([&](auto&&... args) { return func_((decltype(args)&&)args...); },
                           std::move(batch.values_));
    auto& states = batch.states_;
    auto& ranges = batch.ranges_;
    size_t start = 0;
    for (size_t i = 0; i < states.size(); ++i) {
      auto count = ranges[i].second;
      range_t rets_range{start, count};
      states[i]->Notify(rets, rets_range, ranges[i]);
      start += count;
    }
  }

  std::unique_ptr<Batch> batch_;
};

template <typename Sender, typename Scheduler, typename Manager, typename Receiver, typename Func>
struct _Operation<Sender, Scheduler, Manager, Receiver, Func>::type {
  using _context_t = Context<Sender, Scheduler, Manager, Receiver, Func>;
  using _receiver_t = receiver_t<Sender, Scheduler, Manager, Receiver, Func>;
  using _result_t = decltype(std::apply(std::declval<Func>(),
                                        std::declval<completion_signatures_of_t<Sender>>()));

  _context_t* context_;
  connect_result_t<Sender, _receiver_t> op_state_;
  Receiver receiver_;
  _result_t vals_;

  std::atomic<size_t> count_{0};

  template <typename Receiver2>
  type(Sender&& sender, Scheduler scheduler, Manager* manager, Func func, Receiver2&& receiver)
      : context_(CreateContext(manager, std::move(scheduler), std::move(func))),
        op_state_(Connect((Sender &&) sender, _receiver_t{this})),
        receiver_((Receiver2 &&) receiver),
        vals_{} {}

  _context_t* CreateContext(Manager* manager, Scheduler scheduler, Func func) {
    auto& context = manager->context_;
    auto* old = context.load(std::memory_order_acquire);
    if (old) {
      return static_cast<_context_t*>(old);
    } else {
      auto p = std::make_unique<_context_t>(scheduler, manager, std::move(func));
      if (context.compare_exchange_strong(old, p.get(), std::memory_order_release,
                                          std::memory_order_acquire)) {
        // context is filled with p, and now it has the ownership of its value
        return p.release();
      } else {
        // old contains context created by some other thread, p will be destroyed
        return static_cast<_context_t*>(old);
      }
    }
  }

  friend void tag_invoke(start_t, type& self) { Start(self.op_state_); }

  void Notify(_result_t& rets, range_t rets_range, range_t vals_range) {
    Manager::output(rets, rets_range, vals_, vals_range);
    auto count = rets_range.second;
    if (count_.fetch_sub(count, std::memory_order_acq_rel) == count) {  // (count_ -= count) == 0
      SetValue(std::move(receiver_), std::move(vals_));
    }
  }
};

template <typename Sender, typename Scheduler, typename Manager, typename Func>
struct _Sender {
  struct type {
    using _result_t = decltype(std::apply(std::declval<Func>(),
                                          std::declval<completion_signatures_of_t<Sender>>()));

    using value_types = std::tuple<_result_t>;

    Sender sender_;
    Scheduler scheduler_;
    Func func_;
    Manager* manager_;

    template <typename Sender2>
    type(Sender2&& sender, Scheduler scheduler, Manager& manager, Func func)
        : sender_((Sender2 &&) sender),
          scheduler_(std::move(scheduler)),
          manager_(&manager),
          func_(std::move(func)) {}

    template <typename Receiver>
    friend auto tag_invoke(connect_t, type&& self, Receiver&& receiver)
        -> operation_t<Sender, Scheduler, Manager, Receiver, Func> {
      return {std::move(self).sender_, std::move(self).scheduler_, self.manager_,
              std::move(self).func_, (Receiver &&) receiver};
    }
  };
};

template <typename Sender, typename Scheduler, typename Manager, typename Func>
using sender_t = typename _Sender<remove_cvref_t<Sender>, Scheduler, Manager, Func>::type;

template <typename SubmitSch, typename ExecuteSch, typename Sender, typename Manager, typename Func>
auto tag_invoke(dynamic_batch_t, const scheduler_t<SubmitSch, ExecuteSch>& scheduler,
                Sender&& sender, Manager& manager, Func func)
    -> sender_t<Sender, scheduler_t<SubmitSch, ExecuteSch>, Manager, Func> {
  return {(Sender &&) sender, scheduler, manager, std::move(func)};
}

}  // namespace _dynamic_batch_scheduler

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_SCHEDULERS_DYNAMIC_BATCH_SCHEDULER_H_
