//
// Created by li on 2022/3/11.
//

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <variant>

#include "core/mpl/detected.h"
#include "core/utils/formatter.h"
#include "core/value.h"
#include "intrusive_queue.h"

namespace mmdeploy {

template <class T, class E, class U = void>
using _decays_to = std::enable_if_t<std::is_same<std::decay_t<T>, E>::value, U>;

template <class... Ts>
using __decayed_tuple = std::tuple<std::decay_t<Ts>...>;

template <class Fun, class... As>
using __call_result_t = decltype(std::declval<Fun>()(std::declval<As>()...));

template <class F>
struct __conv {
  F f_;
  operator __call_result_t<F>() && { return ((F &&) f_)(); }
};

template <class F>
__conv(F) -> __conv<F>;

template <class _Member, class _Self>
_Member _Self::*__memptr(const _Self&);

template <typename _Self, typename _Member>
using __member_t = decltype((std::declval<_Self>().*__memptr<_Member>(std::declval<_Self>())));

template <class From, class To>
using _copy_cvref_t = __member_t<From, To>;

template <class S, class R>
using connect_result_t = decltype(Connect(std::declval<S>(), std::declval<R>()));

template <class Sched>
using schedule_result_t = decltype(Schedule(std::declval<Sched>()));

template <class Sender>
using _get_completion_scheduler_t = decltype(GetCompletionScheduler(std::declval<Sender>()));

template <class Sender>
inline constexpr auto _has_completion_scheduler =
    detail::is_detected_v<_get_completion_scheduler_t, Sender>;

struct InlineScheduler {
  template <typename R>
  struct _Operation {
    R rec_;
    friend void Start(_Operation& op) noexcept { SetValue((R &&) op.rec_); }
  };

  struct _Sender {
    template <typename R>
    friend auto Connect(_Sender, R&& rec) -> _Operation<std::decay_t<R>> {
      return {(R &&) rec};
    }

    friend InlineScheduler GetCompletionScheduler(_Sender) noexcept { return {}; }
  };

  friend _Sender Schedule(const InlineScheduler) noexcept { return {}; }

  struct _Receiver {
    Value* data_;
    friend void SetValue(_Receiver&& r, Value data) noexcept { *r.data_ = std::move(data); }
  };

  template <class S>
  friend Value SyncWait(InlineScheduler, S&& sender) {
    Value data;
    auto op_state = Connect(((S &&) sender), _Receiver{&data});
    Start(op_state);
    return data;
  }
};

namespace __just {

struct _Sender {
  Value v_;

  template <class R>
  struct _Operation {
    Value v_;
    R r_;
    friend void Start(_Operation& s) noexcept { SetValue(std::move(s.r_), std::move(s.v_)); }
  };

  template <class R>
  friend _Operation<std::decay_t<R>> Connect(_Sender&& s, R&& r) {
    return {std::move(s).v_, std::forward<R>(r)};
  }

  template <class R>
  friend _Operation<std::decay_t<R>> Connect(const _Sender& s, R&& r) {
    return {s.v_, std::forward<R>(r)};
  }
};

}  // namespace __just

inline __just::_Sender Just(Value v) { return {std::move(v)}; }

namespace __on {

template <class Scheduler, class Sender, class Receiver>
struct _Operation;

template <class Scheduler, class Sender, class Receiver>
struct _ReceiverRef {
  _Operation<Scheduler, Sender, Receiver>* op_state_;
  template <class... Args>
  friend void SetValue(_ReceiverRef&& self, Args&&... args) {
    SetValue((Receiver &&) self.op_state_->rcvr_, ((Args &&) args)...);
  }
};

template <class Scheduler, class Sender, class Receiver>
struct _Receiver {
  _Operation<Scheduler, Sender, Receiver>* op_state_;
  using ReceiverRef = _ReceiverRef<Scheduler, Sender, Receiver>;
  friend void SetValue(_Receiver&& self) {
    auto op_state = self.op_state_;
    Start(op_state->data_.template emplace<1>(
        Connect((Sender &&) op_state->sndr_, ReceiverRef{op_state})));
  }
};

template <class Scheduler, class Sender, class Receiver>
struct _Operation {
  using __Receiver = _Receiver<Scheduler, Sender, Receiver>;
  using __ReceiverRef = _ReceiverRef<Scheduler, Sender, Receiver>;

  template <class Sender2, class Receiver2>
  _Operation(Scheduler sched, Sender2&& sndr, Receiver2&& rcvr)
      : data_(std::in_place_index<0>, Connect(Schedule(sched), __Receiver{this})),
        scheduler_(sched),
        sndr_((Sender2 &&) sndr),
        rcvr_((Receiver2 &&) rcvr) {}

  friend void Start(_Operation& self) { Start(std::get<0>(self.data_)); }

  std::variant<connect_result_t<schedule_result_t<Scheduler>, __Receiver>,
               connect_result_t<Sender, __ReceiverRef>>
      data_;
  Scheduler scheduler_;
  Sender sndr_;
  Receiver rcvr_;
};

template <class Scheduler, class Sender>
struct _Sender {
  Scheduler sched_;
  Sender sndr_;

  template <class Receiver>
  using _ReceiverRef = _ReceiverRef<Scheduler, Sender, Receiver>;
  template <class Receiver>
  using _Receiver = _Receiver<Scheduler, Sender, Receiver>;
  template <class Receiver>
  using _Operation = _Operation<Scheduler, Sender, Receiver>;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr) -> _Operation<std::decay_t<Receiver>> {
    return {((Self &&) self).sched_, ((Self &&) self).sndr_, (Receiver &&) rcvr};
  }
};

}  // namespace __on

template <class Scheduler, class Sender>
__on::_Sender<Scheduler, Sender> On(Scheduler&& sched, Sender&& sndr) {
  return {(Scheduler &&) sched, (Sender &&) sndr};
}

namespace __schedule_from {

template <class Scheduler, class CvrefSender, class Receiver>
struct _Operation1;

template <class Scheduler, class CvrefSender, class Receiver>
struct _Receiver1;

template <class Scheduler, class CvrefSender, class Receiver>
struct _Receiver2 {
  _Operation1<Scheduler, CvrefSender, Receiver>* op_state_;

  friend void SetValue(_Receiver2&& self) noexcept {
    SetValue(std::move(self.op_state_->rcvr_), std::move(self.op_state_->data_));
  }
};

template <class Scheduler, class CvrefSender, class Receiver>
struct _Receiver1 {
  using Receiver2 = _Receiver2<Scheduler, CvrefSender, Receiver>;

  _Operation1<Scheduler, CvrefSender, Receiver>* op_state_;

  template <class V, _decays_to<V, Value, bool> = true>
  friend void SetValue(_Receiver1&& self, V&& v) {
    self.op_state_->data_ = (V &&) v;
    auto sndr = Schedule(self.op_state_->sched_);
    self.op_state_->state2_.emplace(Connect(std::move(sndr), Receiver2{self.op_state_}));
    Start(*self.op_state_->state2_);
  }
};

template <class Scheduler, class CvrefSender, class Receiver>
struct _Operation1 {
  using Receiver1 = _Receiver1<Scheduler, CvrefSender, Receiver>;
  using Receiver2 = _Receiver2<Scheduler, CvrefSender, Receiver>;

  Scheduler sched_;
  Receiver rcvr_;
  Value data_;
  connect_result_t<CvrefSender, Receiver1> state1_;
  std::optional<connect_result_t<schedule_result_t<Scheduler>, Receiver2>> state2_;

  template <class R>
  _Operation1(Scheduler sched, CvrefSender&& sndr, R&& rcvr)
      : sched_(sched),
        rcvr_((R &&) rcvr),
        state1_(Connect((CvrefSender &&) sndr, Receiver1{this})) {}

  _Operation1(const _Operation1&) = delete;
  _Operation1(_Operation1&&) noexcept = delete;
  _Operation1& operator=(const _Operation1&) = delete;
  _Operation1& operator=(_Operation1&&) noexcept = delete;

  friend void Start(_Operation1& op_state) noexcept { Start(op_state.state1_); }
};

template <class Scheduler, class Sender>
struct _Sender {
  Scheduler sched_;
  Sender sndr_;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr)
      -> _Operation1<Scheduler, _copy_cvref_t<Self, Sender>, std::decay_t<Receiver>> {
    return {self.sched_, ((Self &&) self).sndr_, (Receiver &&) rcvr};
  }

  friend Scheduler GetCompletionScheduler(const _Sender& self) noexcept { return self.sched_; }
};

}  // namespace __schedule_from

template <class Scheduler, class Sender>
__schedule_from::_Sender<std::decay_t<Scheduler>, std::decay_t<Sender>> ScheduleFrom(
    Scheduler&& sched, Sender&& sndr) {
  return {(Scheduler &&) sched, (Sender &&) sndr};
}

namespace __then {

template <class R, class F>
struct _Receiver {
  R r_;
  F f_;

  template <class... Args>
  friend void SetValue(_Receiver&& self, Args&&... args) {
    SetValue(std::move(self.r_), ((F &&) self.f_)(std::forward<Args>(args)...));
  }
};

template <class S, class F>
struct _Sender {
  S s_;
  F f_;

  template <class Receiver>
  using receiver_t = _Receiver<std::decay_t<Receiver>, F>;

  template <class Self, class R, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, R r) {
    return Connect(((Self &&) self).s_, receiver_t<R>{(R &&) r, (F &&) self.f_});
  }

  template <class Sender = S>
  friend auto GetCompletionScheduler(const _Sender& self) noexcept
      -> decltype(GetCompletionScheduler(std::declval<Sender>())) {
    return GetCompletionScheduler(self.s_);
  }
};

}  // namespace __then

template <class S, class F>
__then::_Sender<std::decay_t<S>, F> Then(S&& s, F f) {
  return {(S &&) s, (F &&) f};
}

namespace __let_value {

template <class T>
using __decay_ref = std::decay_t<T>&;

template <class Fun, class... As>
using __result_sender_t = __call_result_t<Fun, __decay_ref<As>...>;

template <class Sender, class Receiver, class Fun>
struct _Storage {
  using operation_t = connect_result_t<__result_sender_t<Fun, Value>, Receiver>;
  Value data_;
  std::optional<operation_t> op_state3_;
};

template <class Sender, class Receiver, class Fun>
struct _Operation;

template <class Sender, class Receiver, class Fun>
struct _Receiver {
  _Operation<Sender, Receiver, Fun>* op_state_;

  friend void SetValue(_Receiver&& self, Value v) noexcept {
    self.op_state_->storage_.data_ = (Value &&) v;
    Start(self.op_state_->storage_.op_state3_.emplace(__conv{[&]() {
      return Connect(std::move(self.op_state_->fun_)(self.op_state_->storage_.data_),
                     std::move(self.op_state_->rcvr_));
    }}));
  }
};

template <class Sender, class Receiver, class Fun>
struct _Operation {
  using receiver_t = _Receiver<Sender, Receiver, Fun>;

  friend void Start(_Operation& self) noexcept { Start(self.op_state2_); }

  template <class Receiver2>
  _Operation(Sender&& sndr, Receiver2&& rcvr, Fun fun)
      : op_state2_(Connect((Sender &&) sndr, receiver_t{this})),
        rcvr_((Receiver2 &&) rcvr),
        fun_((Fun &&) fun) {}

  connect_result_t<Sender, receiver_t> op_state2_;
  Receiver rcvr_;
  Fun fun_;
  _Storage<Sender, Receiver, Fun> storage_;
};

template <class Sender, class Fun>
struct _Sender {
  template <class Self, class Receiver>
  using operation_t = _Operation<_copy_cvref_t<Self, Sender>, std::decay_t<Receiver>, Fun>;
  template <class Self, class Receiver>
  using receiver_t = _Receiver<_copy_cvref_t<Self, Sender>, std::decay_t<Receiver>, Fun>;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr) -> operation_t<Self, Receiver> {
    return operation_t<Self, Receiver>{((Self &&) self).sndr_, (Receiver &&) rcvr,
                                       ((Self &&) self).fun_};
  }
  Sender sndr_;
  Fun fun_;
};

}  // namespace __let_value

template <class Sender, class Fun,
          typename _Sender = __let_value::_Sender<std::decay_t<Sender>, Fun>>
_Sender LetValue(Sender&& s, Fun&& f) {
  return _Sender{(Sender &&) s, (Fun &&) f};
}

namespace __split {

struct _OperationBase {
  _OperationBase* next_;
  void (*notify_)(_OperationBase*) noexcept;
};

template <class SharedState>
struct _Receiver {
  SharedState& shared_state_;
  friend void SetValue(_Receiver&& recvr, Value v) {
    auto& state = recvr.shared_state_;
    state.data_ = std::move(v);
    state._Notify();
  }
};

template <class Sender>
struct _SharedState {
  std::optional<Value> data_;

  using Receiver = _Receiver<_SharedState>;

  connect_result_t<Sender, Receiver> op_state2_;

  std::atomic<void*> head_{nullptr};

  explicit _SharedState(Sender& sndr) : op_state2_(Connect((Sender &&) sndr, Receiver{*this})) {}

  void _Notify() noexcept {
    void* const completion_state = static_cast<void*>(this);
    void* old = head_.exchange(completion_state, std::memory_order_acq_rel);
    auto* op_state = static_cast<_OperationBase*>(old);

    while (op_state != nullptr) {
      _OperationBase* next = op_state->next_;
      op_state->notify_(op_state);
      op_state = next;
    }
  }
};

template <class Sender, class Receiver>
struct _Operation : _OperationBase {
  Receiver recvr_;
  std::shared_ptr<_SharedState<Sender>> shared_state_;

  _Operation(Receiver&& r, std::shared_ptr<_SharedState<Sender>> shared_state)
      : _OperationBase{nullptr, _Notify},
        recvr_(std::move(r)),
        shared_state_(std::move(shared_state)) {}

  static void _Notify(_OperationBase* self) noexcept {
    auto op = static_cast<_Operation*>(self);
    SetValue((Receiver &&) op->recvr_, *op->shared_state_->data_);
  }

  friend void Start(_Operation& self) {
    auto shared_state = self.shared_state_.get();
    std::atomic<void*>& head = shared_state->head_;
    void* const completion_state = static_cast<void*>(shared_state);
    void* old = head.load(std::memory_order_acquire);

    do {
      if (old == completion_state) {
        self._Notify(&self);
        return;
      }
      self.next_ = static_cast<_OperationBase*>(old);
    } while (!head.compare_exchange_weak(old, static_cast<void*>(&self), std::memory_order_release,
                                         std::memory_order_acquire));

    if (old == nullptr) {
      Start(shared_state->op_state2_);
    }
  }
};

template <class Sender>
struct _Sender {
  using SharedState = _SharedState<Sender>;
  template <class Receiver>
  using Operation = _Operation<Sender, std::decay_t<Receiver>>;

  Sender sndr_;
  std::shared_ptr<SharedState> shared_state_;

  explicit _Sender(Sender sndr)
      : sndr_((Sender &&) sndr), shared_state_{std::make_shared<SharedState>(sndr_)} {}

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& recvr) -> Operation<Receiver> {
    return Operation<Receiver>((std::decay_t<Receiver> &&) recvr, self.shared_state_);
  }
};

}  // namespace __split

template <class Sender>
auto Split(Sender&& sndr) -> __split::_Sender<std::decay_t<Sender>> {
  return __split::_Sender<std::decay_t<Sender>>{(Sender &&) sndr};
}

namespace __when_all {

template <class... Senders>
struct _Sender {
  template <class... _Sndrs>
  explicit _Sender(_Sndrs&&... sndrs) : sndrs_((_Sndrs &&) sndrs...) {}

  template <class CvrefReceiver>
  struct _Operation;

  template <class CvrefReceiver, size_t Index>
  struct _Receiver {
    using WhenAll = _copy_cvref_t<CvrefReceiver, _Sender>;
    using Receiver = std::decay_t<CvrefReceiver>;
    _Operation<CvrefReceiver>* op_state_;

    friend void SetValue(_Receiver&& self, Value v) noexcept {
      self.op_state_->values_[Index] = std::move(v);
      self.op_state_->_Arrive();
    }
  };

  template <class CvrefReceiver>
  struct _Operation {
    using WhenAll = _copy_cvref_t<CvrefReceiver, _Sender>;
    using Receiver = std::decay_t<CvrefReceiver>;
    template <class Sender, size_t Index>
    using _ChildOpState =
        connect_result_t<_copy_cvref_t<WhenAll, Sender>, _Receiver<CvrefReceiver, Index>>;

    using _Indices = std::index_sequence_for<Senders...>;

    template <size_t... Is>
    static auto _ConnectChildren(_Operation* self, WhenAll&& when_all, std::index_sequence<Is...>)
        -> std::tuple<_ChildOpState<Senders, Is>...> {
      return std::tuple<_ChildOpState<Senders, Is>...>{Connect(
          std::get<Is>(((WhenAll &&) when_all).sndrs_), _Receiver<CvrefReceiver, Is>{self})...};
    }

    using _ChildOpStatesTuple =
        decltype(_ConnectChildren(nullptr, std::declval<WhenAll>(), _Indices{}));

    void _Arrive() noexcept {
      if (0 == --count_) {
        _Complete();
      }
    }

    void _Complete() noexcept {
      // just forward array to receiver for now
      SetValue((Receiver &&) recvr_, std::move(values_));
    }

    _Operation(WhenAll&& when_all, Receiver rcvr)
        : child_states_{_ConnectChildren(this, (WhenAll &&) when_all, _Indices{})},
          recvr_((Receiver &&) rcvr),
          values_(sizeof...(Senders)) {}

    friend void Start(_Operation& self) noexcept {
      std::apply([](auto&&... child_ops) noexcept -> void { (Start(child_ops), ...); },
                 self.child_states_);
    }

    _ChildOpStatesTuple child_states_;
    Receiver recvr_;
    std::atomic<size_t> count_{sizeof...(Senders)};
    Value::Array values_;
  };

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr) -> _Operation<_copy_cvref_t<Self, Receiver>> {
    return {(Self &&) self, (Receiver &&) rcvr};
  }

  std::tuple<Senders...> sndrs_;
};

}  // namespace __when_all

template <class... Senders, std::enable_if_t<(sizeof...(Senders) > 0), bool> = true>
auto WhenAll(Senders&&... sndrs) -> __when_all::_Sender<std::decay_t<Senders>...> {
  return __when_all::_Sender<std::decay_t<Senders>...>{(Senders &&) sndrs...};
}

namespace __ensure_started {

struct _OperationBase {
  void (*notify_)(_OperationBase*);
};

template <class Sender>
struct _SharedState {
  // stop_source
  // op_state
  // op_state = connect(s, r)
  Value data_;
};

template <class Sender, class Receiver>
struct _Receiver {
  std::shared_ptr<_SharedState<Sender>> shared_state_;
  friend void SetValue(_Receiver&& self, Value v) {
    assert(self.shared_state_);
    self.shared_state_->data = std::move(v);
    self.shared_state_.reset();
  }
};

template <class Sender, class Receiver>
struct _Operation {
  Receiver rcvr_;
  std::shared_ptr<_SharedState<Sender>> shared_state_;

  friend void Start(_Operation& op_state) {
    if (true) {
      SetValue((Receiver&&)op_state.rcvr_, std::move(op_state.shared_state_->data_));
    }
  }
};

template <class Sender>
struct _Sender {
  std::shared_ptr<_SharedState<Sender>> shared_state_;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr) -> _Operation<Sender, std::decay_t<Receiver>> {
    return {(Receiver &&) rcvr, std::move(self.shared_state_)};
  }
};

}  // namespace __ensure_started

namespace __loop {
class RunLoop;

namespace __impl {

struct _Task {
  virtual void _Execute() noexcept = 0;
  _Task* next_ = nullptr;
};

template <class Receiver>
class _Operation final : _Task {
  friend void Start(_Operation& op_state) noexcept { op_state._Start(); }

  void _Execute() noexcept override { SetValue((Receiver &&) rcvr_); }

  void _Start() noexcept;

  Receiver rcvr_;
  RunLoop* const loop_;

 public:
  template <class _Receiver2>
  explicit _Operation(_Receiver2&& rcvr, RunLoop* loop)
      : rcvr_((_Receiver2 &&) rcvr), loop_(loop) {}
};

}  // namespace __impl

class RunLoop {
  template <class>
  friend class __impl::_Operation;

 public:
  class _Scheduler {
    class _ScheduleTask {
      friend _Scheduler;

      template <class _Receiver>
      friend auto Connect(const _ScheduleTask& self, _Receiver&& rcvr)
          -> __impl::_Operation<std::decay_t<_Receiver>> {
        return {(_Receiver &&) rcvr, self.loop_};
      }

      explicit _ScheduleTask(RunLoop* loop) noexcept : loop_(loop) {}

      RunLoop* const loop_;
    };

    friend RunLoop;

    explicit _Scheduler(RunLoop* loop) noexcept : loop_(loop) {}

   public:
    friend _ScheduleTask Schedule(const _Scheduler& self) noexcept { return self._Schedule(); }

    bool operator==(const _Scheduler& other) const noexcept { return loop_ == other.loop_; }

   private:
    _ScheduleTask _Schedule() const noexcept { return _ScheduleTask{loop_}; }

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

template <class Receiver>
inline void _Operation<Receiver>::_Start() noexcept {
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

namespace __sync_wait {

struct _Receiver {
  Value* data_;
  RunLoop* loop_;
  inline friend void SetValue(_Receiver&& rcvr, Value value) noexcept {
    *rcvr.data_ = std::move(value);
    rcvr.loop_->_Finish();
  }
};

}  // namespace __sync_wait

template <class S, std::enable_if_t<_has_completion_scheduler<S>, bool> = true>
Value SyncWait(S&& sender) {
  auto scheduler = GetCompletionScheduler(sender);
  return SyncWait(scheduler, sender);
}

template <class S>
Value _SyncWaitDefault(S&& sndr) {
  Value data;
  RunLoop loop;

  auto op_state = Connect((S &&) sndr, __sync_wait::_Receiver{&data, &loop});
  Start(op_state);

  loop._Run();

  return data;
}

template <class S, std::enable_if_t<!_has_completion_scheduler<S>, bool> = true>
Value SyncWait(S&& sndr) {
  return _SyncWaitDefault((S &&) sndr);
}

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_
