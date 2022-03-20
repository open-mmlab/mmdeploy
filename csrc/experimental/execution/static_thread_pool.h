//
// Created by li on 2022/3/19.
//

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_STATIC_THREAD_POOL_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_STATIC_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "execution.h"

namespace mmdeploy {

namespace __static_thread_pool {

struct TaskBase {
  TaskBase* next_;
  void (*execute_)(TaskBase*) noexcept;
};

template <class Receiver>
class Operation;

class StaticThreadPool {
  template <class Receiver>
  friend class Operation;

 public:
  StaticThreadPool();
  StaticThreadPool(std::uint32_t thread_count);
  ~StaticThreadPool();

  struct Scheduler {
    bool operator==(const Scheduler&) const;

   private:
    template <class Receiver>
    friend class Operation;

    class Sender {
      template <class Receiver>
      Operation<std::decay_t<Receiver>> MakeOperation_(Receiver&& r) const {}

      template <class Receiver>
      friend Operation<std::decay_t<Receiver>> Connect(Sender s, Receiver&& r) {
        return s.template MakeOperation_((Receiver &&) r);
      }

      friend StaticThreadPool::Scheduler GetCompletionScheduler(Sender s) noexcept {
        return StaticThreadPool::Scheduler{s.pool_};
      }

      friend struct StaticThreadPool::Scheduler;

      explicit Sender(StaticThreadPool& pool) noexcept : pool_(pool) {}

      StaticThreadPool& pool_;
    };

    Sender MakeSender_() const { return Sender{*pool_}; }

    friend Sender Schedule(const Scheduler& s) noexcept { return s.MakeSender_(); }

    friend class StaticThreadPool;

   public:
    explicit Scheduler(StaticThreadPool& pool) noexcept : pool_(&pool) {}

   private:
    StaticThreadPool* pool_;
  };

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
    bool stop_requested_;
  };

  void Run(std::uint32_t index) noexcept;
  void Join() noexcept;

  void Enqueue(TaskBase* task) noexcept;

  std::uint32_t thread_count_;
  std::vector<std::thread> threads_;
  std::vector<ThreadState> thread_states_;
  std::atomic<std::uint32_t> next_thread_;
};

template <class Receiver>
class Operation : TaskBase {
  friend StaticThreadPool::Scheduler::Sender;

  StaticThreadPool& pool_;
  Receiver receiver_;

  explicit Operation(StaticThreadPool& pool, Receiver&& r)
      : pool_(pool), receiver_((Receiver &&) r) {
    this->execute_ = [](TaskBase* t) noexcept {
      auto& op = *static_cast<Operation*>(t);
      SetValue((Receiver &&) op.receiver_);
    };
  }

  void enqueue_(TaskBase* op) const { return pool_.Enqueue(op); }

  friend void Start(Operation& op) noexcept { op.enqueue_(&op); }
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
  const std::uint32_t thread_count = static_cast<std::uint32_t>(threads_.size());
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
  std::unique_lock lock{mutex_, std::try_to_lock};
  if (!lock) {
    return false;
  }
  const bool was_empty = queue_.empty();
  queue_.push_back(task);
  if (was_empty) {
    cv_.notify_one();
  }
  return true;
}

inline void StaticThreadPool::ThreadState::push(TaskBase* task) {
  std::lock_guard lock{mutex_};
  const bool was_empty = queue_.empty();
  queue_.push_back(task);
  if (was_empty) {
    cv_.notify_one();
  }
}

inline void StaticThreadPool::ThreadState::request_stop() {
  std::lock_guard lock{mutex_};
  stop_requested_ = true;
  cv_.notify_one();
}

}  // namespace __static_thread_pool

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_STATIC_THREAD_POOL_H_
