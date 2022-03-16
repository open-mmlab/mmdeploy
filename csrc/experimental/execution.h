//
// Created by li on 2022/3/11.
//

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_

#include <optional>
#include <type_traits>

#include "core/mpl/detected.h"
#include "core/utils/formatter.h"
#include "core/value.h"

namespace mmdeploy {

template <class T, class E, class U = void>
using _decays_to = std::enable_if_t<std::is_same<std::decay_t<T>, E>::value, U>;

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
    friend void SetValue(_Receiver& r, Value data) noexcept { *r.data_ = std::move(data); }
  };

  template <class S>
  friend Value SyncWait(InlineScheduler, S&& sender) {
    Value data;
    _Receiver r{&data};
    auto op_state = Connect(((S &&) sender), r);
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

  template <class R>  //, _decays_to<R, Receiver, bool> = true>
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

  template <typename R>
  friend auto Connect(_Sender&& self, R r) {
    return Connect((_Sender &&) self.s_, _Receiver<R, F>{(R &&) r, (F &&) self.f_});
  }

  template <typename R>
  friend auto Connect(_Sender& self, R r) {
    return Connect(self.s_, _Receiver<R, F>{(R &&) r, (F &&) self.f_});
  }

  friend auto GetCompletionScheduler(const _Sender& self) noexcept
      -> decltype(GetCompletionScheduler(std::declval<S>())) {
    return GetCompletionScheduler(self.s_);
  }
};

}  // namespace __then

template <class S, class F>
__then::_Sender<std::decay_t<S>, F> Then(S&& s, F f) {
  return {(S &&) s, (F &&) f};
}

namespace __sync_wait {}

template <class S>
auto SyncWait(S&& sender)
    -> decltype(SyncWait(GetCompletionScheduler(std::declval<S>()), std::declval<S>())) {
  auto scheduler = GetCompletionScheduler(sender);
  return SyncWait(scheduler, sender);
}

class AbstractScheduler;
class AbstractSender;
class AbstractOperation;
class AbstractReceiver;

template <class _Scheduler>
class TypeErasedScheduler;

template <class _Sender>
class TypeErasedSender;

template <class _Operation>
class TypeErasedOperation;

template <class _Receiver>
class TypeErasedReceiver;

template <class _Sender, class T = std::remove_reference_t<_Sender>>
TypeErasedSender<T>* MakeTypeErasedSender(_Sender&&);

template <class _Scheduler, class T = std::remove_reference_t<_Scheduler>>
TypeErasedScheduler<T>* MakeTypeErasedScheduler(_Scheduler&&);

template <class _Receiver, class T = std::remove_reference_t<_Receiver>>
TypeErasedReceiver<T>* MakeTypeErasedReceiver(_Receiver&&);

// eliminate recursive of type erasers
inline AbstractSender* MakeTypeErasedSender(AbstractSender* s) { return s; }
inline AbstractScheduler* MakeTypeErasedScheduler(AbstractScheduler* s) { return s; }
inline AbstractReceiver* MakeTypeErasedReceiver(AbstractReceiver* r) { return r; }

class AbstractScheduler {
 public:
  virtual ~AbstractScheduler() = default;

  virtual AbstractSender* _Schedule() = 0;
  virtual Value _SyncWait(AbstractSender*) = 0;

  friend AbstractSender* Schedule(AbstractScheduler* self) { return self->_Schedule(); }

  friend Value SyncWait(AbstractScheduler* self, AbstractSender* sender) {
    return self->_SyncWait(sender);
  }
};

class AbstractSender {
 public:
  virtual ~AbstractSender() = default;

  virtual AbstractOperation* _Connect(AbstractReceiver* r) = 0;
  virtual AbstractScheduler* _GetCompletionScheduler() = 0;

  template <class R>
  friend AbstractOperation* Connect(AbstractSender* self, R rcvr) {
    return self->_Connect(MakeTypeErasedReceiver((R &&) rcvr));
  }
  friend AbstractScheduler* GetCompletionScheduler(AbstractSender* self) {
    return self->_GetCompletionScheduler();
  }
};

class AbstractOperation {
 public:
  virtual ~AbstractOperation() = default;

  virtual void _Start() = 0;

  friend void Start(AbstractOperation* self) { self->_Start(); }
};

class AbstractReceiver {
 public:
  virtual ~AbstractReceiver() = default;

  virtual void _SetValue(Value) = 0;

  friend void SetValue(AbstractReceiver* self, Value v) { self->_SetValue(std::move(v)); }

  friend void SetValue(AbstractReceiver* self) { self->_SetValue(Value::kNull); }
};

template <class _Scheduler>
class TypeErasedScheduler : public AbstractScheduler {
 public:
  explicit TypeErasedScheduler(_Scheduler&& scheduler) : scheduler_(std::move(scheduler)) {}
  template <class S>
  explicit TypeErasedScheduler(TypeErasedScheduler<S>*) = delete;
  explicit TypeErasedScheduler(AbstractScheduler*) = delete;

  AbstractSender* _Schedule() override { return MakeTypeErasedSender(Schedule(scheduler_)); }

  Value _SyncWait(AbstractSender* sender) override { return SyncWait(scheduler_, sender); }

 private:
  _Scheduler scheduler_;
};

template <class _Sender>
class TypeErasedSender : public AbstractSender {
 public:
  explicit TypeErasedSender(_Sender&& s) : s_(std::move(s)) {}

  template <class S>
  explicit TypeErasedSender(TypeErasedSender<S>*) = delete;
  explicit TypeErasedSender(AbstractSender*) = delete;

  AbstractOperation* _Connect(AbstractReceiver* r) override {
    // most operation states are non-movable, use copy elision to initialize erased operation
    using _Operation = decltype(Connect(s_, r));
    return new TypeErasedOperation<_Operation>([&] { return Connect(s_, r); });
  }
  AbstractScheduler* _GetCompletionScheduler() override {
    if constexpr (_has_completion_scheduler<_Sender>) {
      auto sched = GetCompletionScheduler(s_);
      return MakeTypeErasedScheduler(std::move(sched));
    } else {
      return nullptr;
    }
  }

 private:
  _Sender s_;
};

template <class _Operation>
class TypeErasedOperation : public AbstractOperation {
 public:
  template <class F>
  explicit TypeErasedOperation(F f) : operation_(f()) {}

  template <class T>
  explicit TypeErasedOperation(TypeErasedOperation<T>*) = delete;
  explicit TypeErasedOperation(AbstractOperation*) = delete;

  void _Start() override { Start(operation_); }

 private:
  _Operation operation_;
};

template <class _Receiver>
class TypeErasedReceiver : public AbstractReceiver {
 public:
  explicit TypeErasedReceiver(_Receiver&& r) : r_(std::move(r)) {}

  template <class R>
  explicit TypeErasedReceiver(TypeErasedReceiver<R>*) = delete;
  explicit TypeErasedReceiver(AbstractReceiver*) = delete;

  void _SetValue(Value v) override {
    if constexpr (detail::is_detected_v<_set_value_t, _Receiver>) {
      SetValue(std::move(r_), std::move(v));
    } else {
      SetValue(std::move(r_));
    }
  }

 private:
  template <class T>
  using _set_value_t = decltype(SetValue(std::declval<T>(), std::declval<Value>()));

  _Receiver r_;
};

template <class _Sender, class T>
TypeErasedSender<T>* MakeTypeErasedSender(_Sender&& sender) {
  return new TypeErasedSender<T>{(_Sender &&) sender};
}

template <class _Scheduler, class T>
TypeErasedScheduler<T>* MakeTypeErasedScheduler(_Scheduler&& scheduler) {
  return new TypeErasedScheduler<T>{(_Scheduler &&) scheduler};
}

template <class _Receiver, class T>
TypeErasedReceiver<T>* MakeTypeErasedReceiver(_Receiver&& receiver) {
  return new TypeErasedReceiver<T>{(_Receiver &&) receiver};
}

}  // namespace mmdeploy

#if __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_value* mmdeploy_value_t;
typedef mmdeploy_value_t (*mmdeploy_invocable_t)(mmdeploy_value_t, void*);

struct mmdeploy_sender;
struct mmdeploy_scheduler;

typedef struct mmdeploy_sender* mmdeploy_sender_t;
typedef struct mmdeploy_scheduler* mmdeploy_scheduler_t;

mmdeploy_scheduler_t mmdeploy_inline_scheduler();

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value);

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler);

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler);

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_invocable_t fn,
                                         void* data);

mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input);

mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t* inputs, int32_t n);

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_
