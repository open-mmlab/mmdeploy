// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_

#include "execution.h"

namespace mmdeploy {

template <class ValueTypes>
class _TypeErasedSender;

template <class ValueTypes>
class _TypeErasedOperation;

template <class ValueTypes>
class _TypeErasedReceiver;

class _TypeErasedScheduler;

#define MMDEPLOY_REQUIRES(...) std::enable_if_t<__VA_ARGS__, int> = 0

template <class ValueTypes>
class _TypeErasedSender {
 public:
  using _Operation = _TypeErasedOperation<ValueTypes>;
  using _Receiver = _TypeErasedReceiver<ValueTypes>;

  using value_type = ValueTypes;

  struct Impl {
    virtual _Operation _Connect(_Receiver) = 0;
    virtual std::unique_ptr<Impl> _Clone() const = 0;
  };

  template <class Sender,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<Sender>, _TypeErasedSender>>>
  _TypeErasedSender(Sender&& sender);

  _TypeErasedSender(_TypeErasedSender&& other) noexcept = default;
  _TypeErasedSender& operator=(_TypeErasedSender&& other) noexcept = default;

  _TypeErasedSender(const _TypeErasedSender& other) : impl_(other.impl_->_Clone()) {}
  _TypeErasedSender& operator=(const _TypeErasedSender& other) {
    impl_ = other.impl_->_Clone();
    return *this;
  }

  template <class Self, class Receiver,
            MMDEPLOY_REQUIRES(std::is_same_v<std::decay_t<Self>, _TypeErasedSender>)>
  friend _Operation Connect(Self&& self, Receiver&& receiver) {
    return self.impl_->_Connect(_TypeErasedReceiver<ValueTypes>((Receiver &&) receiver));
  }

 private:
  std::unique_ptr<Impl> impl_;
};

template <class... Ts>
using TypeErasedSender = _TypeErasedSender<std::tuple<Ts...>>;

template <class Sender>
_TypeErasedSender(Sender&&) -> _TypeErasedSender<completion_signature_for_t<std::decay_t<Sender>>>;

template <class Sender, class ValueTypes = completion_signature_for_t<Sender>>
struct _TypeErasedSenderImpl : _TypeErasedSender<ValueTypes>::Impl {
 public:
  using Base = typename _TypeErasedSender<ValueTypes>::Impl;
  using _Operation = _TypeErasedOperation<ValueTypes>;
  using _Receiver = _TypeErasedReceiver<ValueTypes>;
//  using Type = typename _TypeErasedSender<ValueTypes>::Impl::Type;

  template <class _Sender,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<_Sender>, _TypeErasedSenderImpl>>>
  explicit _TypeErasedSenderImpl(_Sender&& sender) : sender_((_Sender &&) sender) {}

  _Operation _Connect(_Receiver receiver) override {
    return _Operation{[&] { return Connect(std::move(sender_), std::move(receiver)); }};
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
    virtual void _SetValue(ValueTypes) = 0;
  };

  template <class Receiver,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<Receiver>, _TypeErasedReceiver>>>
  explicit _TypeErasedReceiver(Receiver&&);

  template <class... As>
  friend void SetValue(_TypeErasedReceiver&& self, As&&... as) {
    self.impl_->_SetValue(std::forward_as_tuple((As &&) as...));
  }

 private:
  std::unique_ptr<Impl> impl_;
};

template <class Receiver, class ValueTypes>
struct _TypeErasedReceiverImpl : _TypeErasedReceiver<ValueTypes>::Impl {
  void _SetValue(ValueTypes vals) override {
    std::apply(
        [&](auto&&... args) { SetValue((Receiver &&) receiver_, (decltype(args)&&)args...); },
        vals);
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

class _TypeErasedScheduler {
 public:
  struct Impl {
    virtual _TypeErasedSender<std::tuple<>> _Schedule() = 0;
  };

  template <class Scheduler, class = std::enable_if_t<
                                 !std::is_same_v<std::decay_t<Scheduler>, _TypeErasedScheduler>>>
  explicit _TypeErasedScheduler(Scheduler&& sched);

  template <class Self,
            class = std::enable_if_t<std::is_same_v<std::decay_t<Self>, _TypeErasedScheduler>>>
  friend _TypeErasedSender<std::tuple<>> Schedule(Self&& self) {
    return self.impl_->_Schedule();
  }

 private:
  std::shared_ptr<Impl> impl_;
};

template <class Scheduler>
struct _TypeErasedSchedulerImpl : _TypeErasedScheduler::Impl {
  using _SenderType = _TypeErasedSender<std::tuple<>>;
  _SenderType _Schedule() override { return _SenderType{Schedule(scheduler_)}; }

  explicit _TypeErasedSchedulerImpl(Scheduler sched) : scheduler_(std::move(sched)) {}
  Scheduler scheduler_;
};

template <class Scheduler, class>
_TypeErasedScheduler::_TypeErasedScheduler(Scheduler&& scheduler) {
  using _Scheduler = std::decay_t<Scheduler>;
  impl_ = std::make_unique<_TypeErasedSchedulerImpl<_Scheduler>>((Scheduler &&) scheduler);
}

template <class ValueTypes>
class _TypeErasedOperation {
 public:
  struct Impl {
    virtual void _Start() = 0;
  };

  template <class Fun, class = std::enable_if_t<std::is_invocable_v<Fun>>>
  explicit _TypeErasedOperation(Fun&& fun);

  friend void Start(_TypeErasedOperation& op_state) { op_state.impl_->_Start(); }

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

}  // namespace mmdeploy

#if __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_value* mmdeploy_value_t;
typedef mmdeploy_value_t (*mmdeploy_invocable_t)(mmdeploy_value_t, void*);

struct mmdeploy_sender;
struct mmdeploy_scheduler;

typedef mmdeploy_sender (*mmdeploy_kleisli_t)(mmdeploy_value_t, void*);

typedef struct mmdeploy_sender* mmdeploy_sender_t;
typedef struct mmdeploy_scheduler* mmdeploy_scheduler_t;

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_inline_scheduler();

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                                          mmdeploy_scheduler_t scheduler);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input,
                                                      mmdeploy_invocable_t fn, void* context);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_let_value(mmdeploy_sender_t input,
                                                           mmdeploy_kleisli_t kleisli,
                                                           void* context);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t* inputs, int32_t n);

MMDEPLOY_API mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_
