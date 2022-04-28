// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_

#include "execution.h"
#include "execution/schedulers/static_thread_pool.h"

namespace mmdeploy {

namespace _type_erased {

template <class ValueTypes>
class _TypeErasedSender;

template <class ValueTypes>
class _TypeErasedOperation;

template <class ValueTypes>
class _TypeErasedReceiver;

template <class ValueTypes>
class _TypeErasedScheduler;

template <class ValueTypes>
class _TypeErasedSender {
 public:
  using _Operation = _TypeErasedOperation<ValueTypes>;
  using _Receiver = _TypeErasedReceiver<ValueTypes>;

  using value_types = ValueTypes;

  struct Impl {
    virtual ~Impl() = default;
    virtual _Operation _Connect(_Receiver) = 0;
    virtual std::unique_ptr<Impl> _Clone() const = 0;
    virtual void* _GetCompletionSchedulerId() const = 0;
  };

  template <class Sender,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<Sender>, _TypeErasedSender>>>
  /* implicit */ _TypeErasedSender(Sender&& sender);

  _TypeErasedSender(_TypeErasedSender&& other) noexcept = default;
  _TypeErasedSender& operator=(_TypeErasedSender&& other) noexcept = default;

  _TypeErasedSender(const _TypeErasedSender& other) : impl_(other.impl_->_Clone()) {}
  _TypeErasedSender& operator=(const _TypeErasedSender& other) {
    impl_ = other.impl_->_Clone();
    return *this;
  }

  template <class Self, class Receiver, class = _decays_to<Self, _TypeErasedSender>>
  friend _Operation tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
    return self.impl_->_Connect(_TypeErasedReceiver<ValueTypes>((Receiver &&) receiver));
  }

  friend void* GetCompletionSchedulerId(const _TypeErasedSender& self) {
    return self.impl_->_GetCompletionSchedulerId();
  }

 private:
  std::unique_ptr<Impl> impl_;
};

template <class... Ts>
using TypeErasedSender = _TypeErasedSender<std::tuple<Ts...>>;

template <class Sender>
_TypeErasedSender(Sender&&) -> _TypeErasedSender<completion_signatures_of_t<Sender>>;

template <class Sender, class ValueTypes = completion_signatures_of_t<Sender>>
struct _TypeErasedSenderImpl : _TypeErasedSender<ValueTypes>::Impl {
 public:
  using Base = typename _TypeErasedSender<ValueTypes>::Impl;
  using _Operation = _TypeErasedOperation<ValueTypes>;
  using _Receiver = _TypeErasedReceiver<ValueTypes>;

  template <class _Sender,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<_Sender>, _TypeErasedSenderImpl>>>
  explicit _TypeErasedSenderImpl(_Sender&& sender) : sender_((_Sender &&) sender) {}

  _Operation _Connect(_Receiver receiver) override {
    return _Operation{[&] { return Connect(std::move(sender_), std::move(receiver)); }};
  }

  void* _GetCompletionSchedulerId() const override {
    if constexpr (_has_completion_scheduler_v<Sender>) {
      auto sched = GetCompletionScheduler(sender_);
      return GetSchedulerId(sched);
    } else {
      return nullptr;
    }
  }

  std::unique_ptr<Base> _Clone() const override {
    if constexpr (std::is_copy_constructible_v<Sender>) {
      return std::make_unique<_TypeErasedSenderImpl>(sender_);
    } else {
      MMDEPLOY_ERROR("attempt to clone non-copyable sender");
      std::abort();
    }
    return {};
  }

 private:
  Sender sender_;
};

template <class ValueTypes>
template <class Sender, class>
_TypeErasedSender<ValueTypes>::_TypeErasedSender(Sender&& sender) {
  using _Sender = std::decay_t<Sender>;
  impl_ = std::make_unique<_TypeErasedSenderImpl<_Sender>>((Sender &&) sender);
}

template <class ValueTypes>
class _TypeErasedReceiver {
 public:
  struct Impl {
    virtual ~Impl() = default;
    virtual void _SetValue(ValueTypes) = 0;
  };

  template <class Receiver,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<Receiver>, _TypeErasedReceiver>>>
  explicit _TypeErasedReceiver(Receiver&&);

  template <class... As>
  friend void tag_invoke(set_value_t, _TypeErasedReceiver&& self, As&&... as) noexcept {
    self.impl_->_SetValue(std::make_tuple((As &&) as...));
  }

 private:
  std::unique_ptr<Impl> impl_;
};

template <class Receiver, class ValueTypes>
struct _TypeErasedReceiverImpl : _TypeErasedReceiver<ValueTypes>::Impl {
  void _SetValue(ValueTypes vals) override {
    std::apply(
        [&](auto&&... args) noexcept { SetValue(std::move(receiver_), (decltype(args)&&)args...); },
        std::move(vals));
  }
  Receiver receiver_;

  template <class _Receiver>
  explicit _TypeErasedReceiverImpl(_Receiver&& receiver) : receiver_((_Receiver &&) receiver) {}
};

template <class ValueTypes>
template <class Receiver, class>
_TypeErasedReceiver<ValueTypes>::_TypeErasedReceiver(Receiver&& receiver) {
  using _Receiver = std::decay_t<Receiver>;
  impl_ = std::make_unique<_TypeErasedReceiverImpl<_Receiver, ValueTypes>>((Receiver &&) receiver);
}

////////////////////////////////////////////////////////
/// _TypeErasedScheduler

template <class... Ts>
using TypeErasedScheduler = _TypeErasedScheduler<std::tuple<Ts...>>;

template <class>
struct _ThenFn {};

template <class... Ts>
struct _ThenFn<std::tuple<Ts...>> {
  using type = std::function<std::tuple<Ts...>(Ts...)>;
};

template <class>
struct _BulkFn {};

template <class... Ts>
struct _BulkFn<std::tuple<Ts...>> {
  using type = std::function<void(size_t, Ts&...)>;
};

template <class ValueTypes>
class _TypeErasedScheduler {
 public:
  using SenderType = _TypeErasedSender<ValueTypes>;
  using EmptySender = _TypeErasedSender<std::tuple<>>;

  using ThenFun = typename _ThenFn<ValueTypes>::type;
  using BulkFun = typename _BulkFn<ValueTypes>::type;

  struct Impl {
    virtual ~Impl() = default;
    virtual EmptySender _Schedule() = 0;

    virtual void* _GetSchedulerId() = 0;

    virtual SenderType _Transfer(SenderType input, _TypeErasedScheduler sched) {
      return ::mmdeploy::Transfer(std::move(input), std::move(sched));
    }
    virtual SenderType _ScheduleFrom(SenderType) = 0;
    //    virtual SenderType _Then(SenderType input, ThenFun fun) {
    //      return ::mmdeploy::Then(std::move(input), std::move(fun));
    //    }
    // virtual SenderType _LetValue() = 0;
    // virtual SenderType _On(SenderType) = 0;
    virtual SenderType _Bulk(SenderType input, size_t shape, BulkFun fun) {
      return ::mmdeploy::Bulk(std::move(input), shape, std::move(fun));
    }
    // virtual SenderType _Split(SenderType) = 0;
    // virtual SenderType _WhenAll(std::vector<SenderType>) = 0;
    // virtual SenderType _TransferWhenAll(std::vector<SenderType>) = 0;
    // virtual SenderType _EnsureStarted(SenderType) = 0;
    // virtual void _StartDetached(SenderType) = 0;
    // virtual ValueTypes _SyncWait(SenderType) = 0;
  };

  template <class Scheduler, class = std::enable_if_t<
                                 !std::is_same_v<std::decay_t<Scheduler>, _TypeErasedScheduler>>>
  explicit _TypeErasedScheduler(Scheduler&& sched);

  explicit _TypeErasedScheduler(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

  _TypeErasedSender<std::tuple<>> Schedule() { return impl_->_Schedule(); }

  friend void* GetSchedulerId(const _TypeErasedScheduler& self) {
    return self.impl_->_GetSchedulerId();
  }

  SenderType ScheduleFrom(SenderType input) { return impl_->_ScheduleFrom(std::move(input)); }

  SenderType Bulk(SenderType input, size_t shape, BulkFun fun) {
    return impl_->_Bulk(std::move(input), shape, std::move(fun));
  }

 private:
  std::shared_ptr<Impl> impl_;
};

template <class ValueTypes, class Scheduler>
struct _TypeErasedSchedulerImpl : _TypeErasedScheduler<ValueTypes>::Impl {
  using _SenderType = _TypeErasedSender<std::tuple<>>;

  using Base = typename _TypeErasedScheduler<ValueTypes>::Impl;
  using BulkFun = typename _TypeErasedScheduler<ValueTypes>::BulkFun;
  using EmptySender = typename _TypeErasedScheduler<ValueTypes>::EmptySender;
  using SenderType = typename _TypeErasedScheduler<ValueTypes>::SenderType;

  EmptySender _Schedule() override { return _SenderType{Schedule(scheduler_)}; }

  void* _GetSchedulerId() override { return GetSchedulerId(scheduler_); }

  SenderType _ScheduleFrom(SenderType input) override {
    if (GetCompletionSchedulerId(input) == _GetSchedulerId()) {
      return input;
    }
    return Transfer(std::move(input), scheduler_);
  }

  //  SenderType _Transfer(SenderType input, _TypeErasedScheduler<ValueTypes> sched) override {
  //
  //  }

  SenderType _Bulk(SenderType input, size_t shape, BulkFun fun) override {
    assert(GetCompletionSchedulerId(input) == _GetSchedulerId());
    if constexpr (tag_invocable<bulk_t, Scheduler, SenderType, size_t, BulkFun>) {
      return tag_invoke(Bulk, scheduler_, std::move(input), shape, std::move(fun));
    } else {
      return Base::_Bulk(std::move(input), shape, std::move(fun));
    }
  }

  explicit _TypeErasedSchedulerImpl(Scheduler sched) : scheduler_(std::move(sched)) {}
  Scheduler scheduler_;
};

template <class Scheduler, class... Ts>
using TypeErasedSchedulerImpl = _TypeErasedSchedulerImpl<std::tuple<Ts...>, Scheduler>;

template <class ValueTypes>
template <class Scheduler, class>
_TypeErasedScheduler<ValueTypes>::_TypeErasedScheduler(Scheduler&& scheduler) {
  using _Scheduler = std::decay_t<Scheduler>;
  impl_ =
      std::make_unique<_TypeErasedSchedulerImpl<ValueTypes, _Scheduler>>((Scheduler &&) scheduler);
}

template <class ValueTypes>
class _TypeErasedOperation {
 public:
  struct Impl {
    virtual ~Impl() = default;
    virtual void _Start() = 0;
  };

  template <class Fun, class = std::enable_if_t<std::is_invocable_v<Fun>>>
  explicit _TypeErasedOperation(Fun&& fun);

  friend void tag_invoke(start_t, _TypeErasedOperation& op_state) { op_state.impl_->_Start(); }

 private:
  std::unique_ptr<Impl> impl_;
};

template <class... Ts>
using TypeErasedOperation = _TypeErasedOperation<std::tuple<Ts...>>;

template <class Operation, class ValueTypes>
struct _TypeErasedOperationImpl : _TypeErasedOperation<ValueTypes>::Impl {
  virtual void _Start() { Start(operation_); }

  template <class Fun, class = std::enable_if_t<std::is_invocable_v<Fun>>>
  explicit _TypeErasedOperationImpl(Fun&& fun) : operation_{((Fun &&) fun)()} {}

  Operation operation_;
};

template <class ValueTypes>
template <class Fun, class>
_TypeErasedOperation<ValueTypes>::_TypeErasedOperation(Fun&& fun) {
  using _Operation = std::invoke_result_t<Fun>;
  impl_.reset(new _TypeErasedOperationImpl<_Operation, ValueTypes>{(Fun &&) fun});
}

}

using _type_erased::TypeErasedSender;
using _type_erased::TypeErasedScheduler;
using _type_erased::TypeErasedOperation;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_

// Schedule(s)

// Just(...)

// TransferJust(s, ...)
// 1. TransferJust(s, ...)
// 2. Transfer(Just(...), s)

// On(sch, s)
// 1. On(sch, s)
// 2. ...

// Transfer(s, sch)
// 1. Transfer(GetCompletionScheduler(s), s, sch)
// 2. Transfer(s, sch)
// 3. ScheduleFrom(sch, s)

// ScheduleFrom(sch, s)
// 1. ScheduleFrom(sch, s)
// 2. ...

// Then(s, f)
// 1. Then(GetCompletionScheduler(s), s, f)
// 2. Then(s, f)
// 3. ...

// LetValue(s, f)
// 1. LetValue(GetCompletionScheduler(s), s, f)
// 2. LetValue(s, f)
// 3. ...

// Bulk(s, shape, f)
// 1. Bulk(GetCompletionScheduler(s), s, shape, f)
// 2. Bulk(s, shape, f)
// 3. ...

// Split(s)
// 1. Split(GetCompletionScheduler(s), s)
// 2. Split(s)
// 3. ...

// WhenAll(s...)
// 1. WhenAll(s...)
// 2. ...

// TransferWhenAll(sch, s...)
// 1. TransferWhenAll(sch, s...)
// 2. Transfer(WhenAll(s...), sch)

// EnsureStarted(s)
// 1. EnsureStarted(GetCompletionScheduler(s), s)
// 2. EnsureStarted(s)
// 3. ...

// StartDetached(s)
// 1. StartDetached(GetCompletionScheduler(s), s)
// 2. StartDetached(s)
// 3. ...

// SyncWait(s)
// 1. SyncWait(GetCompletionScheduler(s), s)
// 2. SyncWait(s)
// 3. ...

// Execute(sch, f)
// 1. Execute(sch, f)
// 2. StartDetached(Then(Schedule(sch), f))

///////////////////////////////////////////////////////////////////////////////////////////

// monadic
// let_value :: Sender a -> (a -> Sender b) -> Sender b

// applicative
// ???       :: Sender a -> Sender (a -> b) -> Sender b
