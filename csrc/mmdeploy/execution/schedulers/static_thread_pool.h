// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/examples/schedulers/static_thread_pool.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_STATIC_THREAD_POOL_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_STATIC_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "intrusive_queue.h"
#include "mmdeploy/execution/execution.h"

namespace mmdeploy {

namespace __static_thread_pool {

struct TaskBase {
  TaskBase* next_;
  void (*execute_)(TaskBase*) noexcept;
};

template <typename Receiver>
struct _Operation {
  struct type;
};
template <typename Receiver>
using operation_t = typename _Operation<remove_cvref_t<Receiver>>::type;

class StaticThreadPool;

struct Scheduler {
  template <typename Receiver>
  friend struct _Operation;

  struct Sender {
    using value_types = std::tuple<>;

    template <typename Receiver>
    operation_t<Receiver> MakeOperation(Receiver&& r) const {
      return {pool_, (Receiver &&) r};
    }

    template <typename Receiver>
    friend operation_t<Receiver> tag_invoke(connect_t, Sender s, Receiver&& r) {
      return s.MakeOperation((Receiver &&) r);
    }

    friend auto tag_invoke(get_completion_scheduler_t, const Sender& sender) noexcept -> Scheduler {
      return Scheduler{sender.pool_};
    }

    friend struct Scheduler;

    explicit Sender(StaticThreadPool& pool) noexcept : pool_(pool) {}

    StaticThreadPool& pool_;
  };

  Sender MakeSender_() const { return Sender{*pool_}; }

  friend class StaticThreadPool;

 public:
  explicit Scheduler(StaticThreadPool& pool) noexcept : pool_(&pool) {}

  friend bool operator==(Scheduler a, Scheduler b) noexcept { return a.pool_ == b.pool_; }

  friend bool operator!=(Scheduler a, Scheduler b) noexcept { return a.pool_ != b.pool_; }

  friend Sender tag_invoke(schedule_t, const Scheduler& self) noexcept {
    return self.MakeSender_();
  }

 private:
  StaticThreadPool* pool_{nullptr};
};

class StaticThreadPool {
  template <typename Receiver>
  friend struct _Operation;

 public:
  StaticThreadPool();
  explicit StaticThreadPool(std::uint32_t thread_count);
  ~StaticThreadPool();

  Scheduler GetScheduler() noexcept { return Scheduler{*this}; }

  void RequestStop() noexcept;

 private:
  class ThreadState {
   public:
    TaskBase* try_pop();
    TaskBase* pop();
    bool try_push(TaskBase* task);
    void push(TaskBase* task);
    void request_stop();

   private:
    std::mutex mutex_;
    std::condition_variable cv_;
    intrusive_queue<&TaskBase::next_> queue_;
    bool stop_requested_{false};
  };

  void Run(std::uint32_t index) noexcept;
  void Join() noexcept;

  void Enqueue(TaskBase* task) noexcept;

  std::uint32_t thread_count_;
  std::vector<std::thread> threads_;
  std::vector<ThreadState> thread_states_;
  std::atomic<std::uint32_t> next_thread_;
};

template <typename Receiver>
struct _Operation<Receiver>::type : TaskBase {
  friend Scheduler::Sender;

  StaticThreadPool& pool_;
  Receiver receiver_;

  type(StaticThreadPool& pool, Receiver&& r) : TaskBase{}, pool_(pool), receiver_((Receiver &&) r) {
    this->execute_ = [](TaskBase* t) noexcept {
      auto& op = *static_cast<type*>(t);
      SetValue((Receiver &&) op.receiver_);
    };
  }

  void enqueue_(TaskBase* op) const { return pool_.Enqueue(op); }

  friend void tag_invoke(start_t, type& op) noexcept { op.enqueue_(&op); }
};

inline StaticThreadPool::StaticThreadPool()
    : StaticThreadPool(std::thread::hardware_concurrency()) {}

inline StaticThreadPool::StaticThreadPool(std::uint32_t thread_count)
    : thread_count_(thread_count), thread_states_(thread_count), next_thread_(0) {
  assert(thread_count_ > 0);

  threads_.reserve(thread_count_);

  try {
    for (std::uint32_t i = 0; i < thread_count_; ++i) {
      threads_.emplace_back([this, i] { Run(i); });
    }
  } catch (...) {
    RequestStop();
    Join();
    throw;
  }
}

inline StaticThreadPool::~StaticThreadPool() {
  RequestStop();
  Join();
}

inline void StaticThreadPool::RequestStop() noexcept {
  for (auto& state : thread_states_) {
    state.request_stop();
  }
}

inline void StaticThreadPool::Run(std::uint32_t index) noexcept {
  while (true) {
    TaskBase* task = nullptr;
    for (std::uint32_t i = 0; i < thread_count_; ++i) {
      auto queue_index = (index + i) < thread_count_ ? (index + i) : (index + i - thread_count_);
      auto& state = thread_states_[queue_index];
      task = state.try_pop();
      if (task != nullptr) {
        break;
      }
    }
    if (task == nullptr) {
      task = thread_states_[index].pop();
      if (task == nullptr) {
        return;
      }
    }
    task->execute_(task);
  }
}

inline void StaticThreadPool::Join() noexcept {
  for (auto& t : threads_) {
    t.join();
  }
  threads_.clear();
}

inline void StaticThreadPool::Enqueue(TaskBase* task) noexcept {
  const auto thread_count = static_cast<std::uint32_t>(threads_.size());
  const std::uint32_t start_index =
      next_thread_.fetch_add(1, std::memory_order_relaxed) % thread_count;
  for (std::uint32_t i = 0; i < thread_count; ++i) {
    const auto index =
        (start_index + i) < thread_count ? (start_index + i) : (start_index + i - thread_count);
    if (thread_states_[index].try_push(task)) {
      return;
    }
  }
  thread_states_[start_index].push(task);
}

inline TaskBase* StaticThreadPool::ThreadState::try_pop() {
  std::unique_lock lock{mutex_, std::try_to_lock};
  if (!lock || queue_.empty()) {
    return nullptr;
  }
  return queue_.pop_front();
}

inline TaskBase* StaticThreadPool::ThreadState::pop() {
  std::unique_lock lock{mutex_};
  while (queue_.empty()) {
    if (stop_requested_) {
      return nullptr;
    }
    cv_.wait(lock);
  }
  return queue_.pop_front();
}

inline bool StaticThreadPool::ThreadState::try_push(TaskBase* task) {
  bool was_empty{};
  {
    std::unique_lock lock{mutex_, std::try_to_lock};
    if (!lock) {
      return false;
    }
    was_empty = queue_.empty();
    queue_.push_back(task);
  }
  if (was_empty) {
    cv_.notify_one();
  }
  return true;
}

inline void StaticThreadPool::ThreadState::push(TaskBase* task) {
  bool was_empty{};
  {
    std::lock_guard lock{mutex_};
    was_empty = queue_.empty();
    queue_.push_back(task);
  }
  if (was_empty) {
    cv_.notify_one();
  }
}

inline void StaticThreadPool::ThreadState::request_stop() {
  {
    std::lock_guard lock{mutex_};
    stop_requested_ = true;
  }
  cv_.notify_one();
}

namespace __bulk {

template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
struct _Operation {
  struct type;
};
template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
using operation_t = typename _Operation<CvrefSender, Shape, Func, Receiver>::type;

template <typename Receiver, typename Shape, typename Func, typename Tuple>
struct _Receiver {
  struct type;
};
template <typename Receiver, typename Shape, typename Func, typename Tuple>
using receiver_t = typename _Receiver<remove_cvref_t<Receiver>, Shape, Func, Tuple>::type;

template <typename Receiver, typename Shape, typename Func, typename Tuple>
struct _Receiver<Receiver, Shape, Func, Tuple>::type {
  struct State {
    Receiver receiver_;
    Shape shape_;
    Func func_;
    std::optional<Tuple> values_;
    Scheduler scheduler_;
    std::atomic<Shape> count_;
  };

  std::shared_ptr<State> state_;

  type(Receiver&& receiver, Shape shape, Func func, Scheduler scheduler)
      : state_(new State{(Receiver &&) receiver, shape, (Func &&) func, std::nullopt, scheduler,
                         shape}) {}

  template <typename... As>
  friend void tag_invoke(set_value_t, type&& self, As&&... as) noexcept {
    auto& state = self.state_;
    state->values_.emplace((As &&) as...);
    for (Shape index = {}; index < state->shape_; ++index) {
      StartDetached(Then(Schedule(state->scheduler_), [state, index] {
        std::apply([&](auto&... vals) { state->func_(index, vals...); }, state->values_.value());
        if (0 == --state->count_) {
          std::apply(
              [&](auto&... vals) { SetValue(std::move(state->receiver_), std::move(vals)...); },
              state->values_.value());
        }
        return 0;
      }));
    }
  }
};

template <typename Sender, typename Shape, typename Func>
struct _Sender {
  struct type;
};
template <typename Sender, typename Shape, typename Func>
using sender_t = typename _Sender<remove_cvref_t<Sender>, remove_cvref_t<Shape>, Func>::type;

template <typename Sender, typename Shape, typename Func>
struct _Sender<Sender, Shape, Func>::type {
  using value_types = completion_signatures_of_t<Sender>;
  template <typename Receiver>
  using _receiver_t = receiver_t<Receiver, Shape, Func, value_types>;

  Sender sender_;
  Scheduler scheduler_;
  Shape shape_;
  Func func_;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
    return Connect(((Self &&) self).sender_,
                   _receiver_t<Receiver>{(Receiver &&) receiver, ((Self &&) self).shape_,
                                         ((Self &&) self).func_, ((Self &&) self).scheduler_});
  }
};

}  // namespace __bulk

template <typename Sender, typename Shape, typename Func>
__bulk::sender_t<Sender, Shape, Func> tag_invoke(bulk_t, Scheduler scheduler, Sender&& sender,
                                                 Shape&& shape, Func&& func) {
  return {(Sender &&) sender, scheduler, (Shape &&) shape, (Func &&) func};
}

}  // namespace __static_thread_pool

using __static_thread_pool::StaticThreadPool;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_STATIC_THREAD_POOL_H_
