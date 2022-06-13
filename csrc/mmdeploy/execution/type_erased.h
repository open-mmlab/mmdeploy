// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_

#include "mmdeploy/execution/execution.h"

// ! DO NOT INCLUDE THIS FILE DIRECTLY IF SPECIALIZATION OF `capture_completion_scheduler` IS
// NEEDED, ALL TRANSLATION UNITS MUST SEE THE SAME SPECIALIZATION

namespace mmdeploy {

namespace _capture_completion_scheduler {

template <typename ValueTypes>
struct capture_completion_scheduler : std::false_type {};

}  // namespace _capture_completion_scheduler

using _capture_completion_scheduler::capture_completion_scheduler;

template <typename ValueTypes>
inline constexpr bool _capture_completion_scheduler_v =
    capture_completion_scheduler<ValueTypes>::value;

namespace _type_erased {

template <typename ValueTypes>
class _TypeErasedSender;

class _TypeErasedOperation;

template <typename ValueTypes>
class _TypeErasedReceiver;

template <typename ValueTypes>
class _TypeErasedScheduler;

struct _unit {};

template <typename>
struct _ThenFn {
  using type = _unit;
};
template <typename T>
struct _ThenFn<std::tuple<T>> {
  using type = std::function<T(T)>;
};
template <>
struct _ThenFn<std::tuple<>> {
  using type = std::function<void()>;
};
template <typename ValueTypes>
using _then_fn_t = typename _ThenFn<ValueTypes>::type;

template <typename>
struct _BulkFn {};
template <typename... Ts>
struct _BulkFn<std::tuple<Ts...>> {
  using type = std::function<void(size_t, Ts&...)>;
};
template <typename ValueTypes>
using _bulk_fn_t = typename _BulkFn<ValueTypes>::type;

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

using TypeErasedOperation = _TypeErasedOperation;

class _TypeErasedOperation {
 public:
  struct Impl {
    virtual ~Impl() = default;
    virtual void _Start() = 0;
  };

  template <typename Fun, typename = std::enable_if_t<std::is_invocable_v<Fun>>>
  explicit _TypeErasedOperation(Fun&& fun);

  friend void tag_invoke(start_t, _TypeErasedOperation& op_state) { op_state.impl_->_Start(); }

 private:
  std::unique_ptr<Impl> impl_;
};

template <typename Operation>
struct _TypeErasedOperationImpl : _TypeErasedOperation::Impl {
  virtual void _Start() { Start(operation_); }

  template <typename Fun, typename = std::enable_if_t<std::is_invocable_v<Fun>>>
  explicit _TypeErasedOperationImpl(Fun&& fun) : operation_{((Fun &&) fun)()} {}

  Operation operation_;
};

template <typename Fun, typename>
_TypeErasedOperation::_TypeErasedOperation(Fun&& fun) {
  using _Operation = std::invoke_result_t<Fun>;
  impl_.reset(new _TypeErasedOperationImpl<_Operation>{(Fun &&) fun});
}

///////////////////////////////////////////////////////////////////////////////
// Sender
///////////////////////////////////////////////////////////////////////////////

template <typename SenderType>
class _TypeErasedSenderAdapter {
 public:
  using value_types = typename SenderType::value_types;

  explicit _TypeErasedSenderAdapter(SenderType sender) : sender_(std::move(sender)) {}

  template <typename Self, typename Receiver, _decays_to<Self, _TypeErasedSenderAdapter, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
    return Connect(((Self &&) self).sender_, (Receiver &&) receiver);
  }

 private:
  SenderType sender_;
};

template <typename SenderType>
_TypeErasedSenderAdapter(SenderType &&)->_TypeErasedSenderAdapter<remove_cvref_t<SenderType>>;

namespace _expose {

template <typename ValueTypes>
struct _Sender {
  using value_types = ValueTypes;

  _TypeErasedSender<ValueTypes> sender_;

  template <typename Self, typename Receiver, _decays_to<Self, _Sender, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
    return Connect(((Self &&) self).sender_, (Receiver &&) receiver);
  }

  friend auto tag_invoke(get_completion_scheduler_t, const _Sender& self) noexcept {
    return self.sender_._GetCompletionScheduler();
  }
};

}  // namespace _expose

template <typename ValueTypes>
class _TypeErasedSender {
 public:
  using _Operation = _TypeErasedOperation;
  using _Receiver = _TypeErasedReceiver<ValueTypes>;
  using _Scheduler = _TypeErasedScheduler<ValueTypes>;
  using value_types = ValueTypes;

  struct Impl {
    virtual ~Impl() = default;
    virtual _Operation _Connect(_Receiver) = 0;
    virtual std::unique_ptr<Impl> _Clone() const = 0;
    virtual _Scheduler _GetCompletionScheduler() const = 0;
  };

  _TypeErasedSender(_TypeErasedSender&& other) noexcept = default;
  _TypeErasedSender& operator=(_TypeErasedSender&& other) noexcept = default;

  _TypeErasedSender(const _TypeErasedSender& other) : impl_(other.impl_->_Clone()) {}
  _TypeErasedSender& operator=(const _TypeErasedSender& other) {
    impl_ = other.impl_->_Clone();
    return *this;
  }

  _Scheduler _GetCompletionScheduler() const { return impl_->_GetCompletionScheduler(); }

  template <typename Self, typename Receiver,
            std::enable_if_t<std::is_same_v<_TypeErasedSender, remove_cvref_t<Self>>, int> = 0>
  friend _Operation tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
    return self.impl_->_Connect(_TypeErasedReceiver<ValueTypes>((Receiver &&) receiver));
  }

  using SenderType = _TypeErasedSender;

  friend _expose::_Sender<ValueTypes> tag_invoke(transfer_t, SenderType input,
                                                 _Scheduler scheduler) {
    auto sched = input.impl_->_GetCompletionScheduler();
    return _expose::_Sender<ValueTypes>{
        tag_invoke(transfer_t{}, sched, std::move(input), std::move(scheduler))};
  }

  template <
      typename Sender,
      typename = std::enable_if_t<
          !std::is_same_v<remove_cvref_t<Sender>, _TypeErasedSender> &&
          !std::is_same_v<remove_cvref_t<Sender>, _TypeErasedSenderAdapter<_TypeErasedSender>>>>
  /* implicit */ _TypeErasedSender(Sender&& sender);

 private:
  std::unique_ptr<Impl> impl_;
};

template <typename... Ts>
using TypeErasedSender = _TypeErasedSender<std::tuple<Ts...>>;

template <typename Sender>
_TypeErasedSender(Sender &&)->_TypeErasedSender<completion_signatures_of_t<Sender>>;

template <typename Sender, typename ValueTypes = completion_signatures_of_t<Sender>>
struct _TypeErasedSenderImpl : _TypeErasedSender<ValueTypes>::Impl {
 public:
  using Base = typename _TypeErasedSender<ValueTypes>::Impl;
  using _Operation = _TypeErasedOperation;
  using _Receiver = _TypeErasedReceiver<ValueTypes>;

  template <typename _Sender, typename = std::enable_if_t<
                                  !std::is_same_v<std::decay_t<_Sender>, _TypeErasedSenderImpl>>>
  explicit _TypeErasedSenderImpl(_Sender&& sender) : sender_((_Sender &&) sender) {}

  _TypeErasedOperation _Connect(_Receiver receiver) override;

  _TypeErasedScheduler<ValueTypes> _GetCompletionScheduler() const override {
    //    static_assert(
    //        !std::is_same_v<ValueTypes, std::tuple<mmdeploy::Value>> ||
    //        (_capture_completion_scheduler_v<ValueTypes> && _has_completion_scheduler_v<Sender>));
    if constexpr (_capture_completion_scheduler_v<ValueTypes> &&
                  _has_completion_scheduler_v<Sender>) {
      return GetCompletionScheduler(sender_);
    } else {
      return _TypeErasedScheduler<ValueTypes>{
          std::make_shared<typename _TypeErasedScheduler<ValueTypes>::Impl>()};
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
template <typename Sender, typename ValueTypes>
_TypeErasedOperation _TypeErasedSenderImpl<Sender, ValueTypes>::_Connect(
    _TypeErasedSenderImpl::_Receiver receiver) {
  return _Operation{[&] { return Connect(std::move(sender_), std::move(receiver)); }};
}

template <typename ValueTypes>
template <typename Sender, typename>
_TypeErasedSender<ValueTypes>::_TypeErasedSender(Sender&& sender) {
  using _Sender = remove_cvref_t<Sender>;
  impl_ = std::make_unique<_TypeErasedSenderImpl<_Sender>>((Sender &&) sender);
}

///////////////////////////////////////////////////////////////////////////////
// Receiver
///////////////////////////////////////////////////////////////////////////////

template <typename ValueTypes>
class _TypeErasedReceiver {
 public:
  struct Impl {
    virtual ~Impl() = default;
    virtual void _SetValue(ValueTypes) = 0;
  };

  template <typename Receiver, typename = std::enable_if_t<
                                   !std::is_same_v<std::decay_t<Receiver>, _TypeErasedReceiver>>>
  explicit _TypeErasedReceiver(Receiver&&);

  template <typename... As>
  friend void tag_invoke(set_value_t, _TypeErasedReceiver&& self, As&&... as) noexcept {
    self.impl_->_SetValue(std::make_tuple((As &&) as...));
  }

 private:
  std::unique_ptr<Impl> impl_;
};

template <typename Receiver, typename ValueTypes>
struct _TypeErasedReceiverImpl : _TypeErasedReceiver<ValueTypes>::Impl {
  void _SetValue(ValueTypes vals) override {
    std::apply(
        [&](auto&&... args) noexcept { SetValue(std::move(receiver_), (decltype(args)&&)args...); },
        std::move(vals));
  }
  Receiver receiver_;

  template <typename _Receiver>
  explicit _TypeErasedReceiverImpl(_Receiver&& receiver) : receiver_((_Receiver &&) receiver) {}
};

template <typename ValueTypes>
template <typename Receiver, typename>
_TypeErasedReceiver<ValueTypes>::_TypeErasedReceiver(Receiver&& receiver) {
  using _Receiver = std::decay_t<Receiver>;
  impl_ = std::make_unique<_TypeErasedReceiverImpl<_Receiver, ValueTypes>>((Receiver &&) receiver);
}

///////////////////////////////////////////////////////////////////////////////
// Scheduler
///////////////////////////////////////////////////////////////////////////////

template <typename... Ts>
using TypeErasedScheduler = _TypeErasedScheduler<std::tuple<Ts...>>;

template <typename ValueTypes>
class _TypeErasedScheduler {
 public:
  using SenderType = _TypeErasedSender<ValueTypes>;
  using SenderAdapterType = _TypeErasedSenderAdapter<SenderType>;
  using EmptySenderType = _TypeErasedSender<std::tuple<>>;

  using ThenFun = typename _ThenFn<ValueTypes>::type;
  using BulkFun = typename _BulkFn<ValueTypes>::type;

  struct Impl {
    virtual ~Impl() = default;
    virtual EmptySenderType _Schedule() { return Just(); }
    virtual SenderType _Transfer(SenderAdapterType input, _TypeErasedScheduler sched) {
      return ::mmdeploy::Transfer(std::move(input), std::move(sched));
    }
    virtual SenderType _Bulk(SenderAdapterType input, size_t shape, BulkFun fun) {
      return ::mmdeploy::Bulk(std::move(input), shape, std::move(fun));
    }
    virtual SenderType _DynamicBatch(SenderAdapterType input, dynamic_batch_t::context_t& context,
                                     ThenFun fun) {
      if constexpr (!std::is_same_v<ThenFun, _unit>) {
        return ::mmdeploy::DynamicBatch(std::move(input), nullptr, std::move(fun));
      } else {
        std::abort();
      }
    }
    // virtual SenderType _ScheduleFrom(SenderType) = 0;
    // virtual SenderType _Then(SenderType input, ThenFun fun) = 0;
    // virtual SenderType _LetValue() = 0;
    // virtual SenderType _On(SenderType) = 0;
    // virtual SenderType _Split(SenderType) = 0;
    // virtual SenderType _WhenAll(std::vector<SenderType>) = 0;
    // virtual SenderType _TransferWhenAll(std::vector<SenderType>) = 0;
    // virtual SenderType _EnsureStarted(SenderType) = 0;
    // virtual void _StartDetached(SenderType) = 0;
    // virtual ValueTypes _SyncWait(SenderType) = 0;
  };

  template <typename Scheduler, typename = std::enable_if_t<
                                    !std::is_same_v<std::decay_t<Scheduler>, _TypeErasedScheduler>>>
  explicit _TypeErasedScheduler(Scheduler&& sched);

  explicit _TypeErasedScheduler(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
    assert(impl_);
  }

  friend EmptySenderType tag_invoke(schedule_t, const _TypeErasedScheduler& self) {
    return self.impl_->_Schedule();
  }

  friend SenderType tag_invoke(transfer_t, const _TypeErasedScheduler& self, SenderType input,
                               _TypeErasedScheduler other) {
    if (self.impl_ == other.impl_) {
      return std::move(input);
    } else {
      return self.impl_->_Transfer(SenderAdapterType{std::move(input)}, std::move(other));
    }
  }

  friend SenderType tag_invoke(bulk_t, const _TypeErasedScheduler& self, SenderType input,
                               size_t shape, BulkFun fun) {
    return self.impl_->_Bulk(SenderAdapterType{std::move(input)}, shape, std::move(fun));
  }

  friend SenderType tag_invoke(dynamic_batch_t, const _TypeErasedScheduler& self, SenderType input,
                               dynamic_batch_t::context_t& context, ThenFun fun) {
    return self.impl_->_DynamicBatch(SenderAdapterType{std::move(input)}, context, std::move(fun));
  }

 private:
  std::shared_ptr<Impl> impl_;
};

template <typename ValueTypes, typename Scheduler>
struct _TypeErasedSchedulerImpl : _TypeErasedScheduler<ValueTypes>::Impl {
  using _SenderType = _TypeErasedSender<std::tuple<>>;

  using Base = typename _TypeErasedScheduler<ValueTypes>::Impl;
  using BulkFun = typename _TypeErasedScheduler<ValueTypes>::BulkFun;
  using ThenFun = typename _TypeErasedScheduler<ValueTypes>::ThenFun;
  using VoidSenderType = typename _TypeErasedScheduler<ValueTypes>::EmptySenderType;
  using SenderType = typename _TypeErasedScheduler<ValueTypes>::SenderType;
  using SenderAdapterType = _TypeErasedSenderAdapter<SenderType>;

  VoidSenderType _Schedule() override { return VoidSenderType{Schedule(scheduler_)}; }

  SenderType _Transfer(SenderAdapterType input, _TypeErasedScheduler<ValueTypes> sched) override {
    if constexpr (tag_invocable<transfer_t, Scheduler, SenderType,
                                _TypeErasedScheduler<ValueTypes>>) {
      return tag_invoke(transfer_t{}, scheduler_, std::move(input), std::move(sched));
    } else {
      return Base::_Transfer(std::move(input), std::move(sched));
    }
  }

  SenderType _Bulk(SenderAdapterType input, size_t shape, BulkFun fun) override {
    if constexpr (tag_invocable<bulk_t, Scheduler, SenderType, size_t, BulkFun>) {
      return tag_invoke(bulk_t{}, scheduler_, std::move(input), shape, std::move(fun));
    } else {
      return Base::_Bulk(std::move(input), shape, std::move(fun));
    }
  }

  SenderType _DynamicBatch(SenderAdapterType input, dynamic_batch_t::context_t& context,
                           ThenFun fun) override {
    if constexpr (tag_invocable<dynamic_batch_t, Scheduler, SenderAdapterType,
                                dynamic_batch_t::context_t&, ThenFun>) {
      return tag_invoke(dynamic_batch_t{}, scheduler_, std::move(input), context, std::move(fun));
    } else {
      return Base::_DynamicBatch(std::move(input), context, std::move(fun));
    }
  }

  explicit _TypeErasedSchedulerImpl(Scheduler sched) : scheduler_(std::move(sched)) {}
  Scheduler scheduler_;
};

template <typename Scheduler, typename... Ts>
using TypeErasedSchedulerImpl = _TypeErasedSchedulerImpl<std::tuple<Ts...>, Scheduler>;

template <typename ValueTypes>
template <typename Scheduler, typename>
_TypeErasedScheduler<ValueTypes>::_TypeErasedScheduler(Scheduler&& scheduler) {
  using _Scheduler = std::decay_t<Scheduler>;
  impl_ =
      std::make_unique<_TypeErasedSchedulerImpl<ValueTypes, _Scheduler>>((Scheduler &&) scheduler);
}

struct type_erase_t {
  template <typename Sender, std::enable_if_t<_is_sender<Sender>, int> = 0>
  auto operator()(Sender&& sender) const {
    return _TypeErasedSender((Sender &&) sender);
  }
  _BinderBack<type_erase_t> operator()() const { return {{}, {}, {}}; }
};

}  // namespace _type_erased

using _type_erased::type_erase_t;
inline constexpr type_erase_t TypeErase{};

using _type_erased::TypeErasedOperation;
using _type_erased::TypeErasedScheduler;
using _type_erased::TypeErasedSender;

// TODO move the specialization somewhere else in a consistent way
class Value;

namespace _capture_completion_scheduler {
template <>
struct capture_completion_scheduler<std::tuple<Value>> : std::true_type {};
}  // namespace _capture_completion_scheduler

}  // namespace mmdeploy
//

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_ERASED_H_
