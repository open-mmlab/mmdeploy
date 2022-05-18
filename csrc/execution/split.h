// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SPLIT_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SPLIT_H_

#include "closure.h"
#include "concepts.h"
#include "utility.h"

namespace mmdeploy {

namespace __split {

template <typename SharedState>
struct _Receiver {
  struct type;
};
template <typename SharedState>
using receiver_t = typename _Receiver<SharedState>::type;

struct _OperationBase {
  _OperationBase* next_;
  void (*notify_)(_OperationBase*) noexcept;
};

template <typename SharedState>
struct _Receiver<SharedState>::type {
  SharedState& shared_state_;

  template <typename... As>
  friend void tag_invoke(set_value_t, type&& self, As&&... as) noexcept {
    auto& state = self.shared_state_;
    state.data_.emplace((As &&) as...);
    state._Notify();
  }
};

template <typename Sender>
struct _SharedState {
  std::optional<completion_signatures_of_t<Sender>> data_;

  using Receiver = receiver_t<_SharedState>;

  connect_result_t<Sender, Receiver> op_state2_;

  std::atomic<void*> head_{nullptr};

  explicit _SharedState(Sender& sender)
      : op_state2_(Connect((Sender &&) sender, Receiver{*this})) {}

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

template <typename Sender, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Sender, typename Receiver>
using operation_t = typename _Operation<Sender, remove_cvref_t<Receiver>>::type;

template <typename Sender, typename Receiver>
struct _Operation<Sender, Receiver>::type : _OperationBase {
  Receiver receiver_;
  std::shared_ptr<_SharedState<Sender>> shared_state_;

  type(Receiver&& receiver, std::shared_ptr<_SharedState<Sender>> shared_state)
      : _OperationBase{nullptr, _Notify},
        receiver_(std::move(receiver)),
        shared_state_(std::move(shared_state)) {}

  static void _Notify(_OperationBase* self) noexcept {
    auto op = static_cast<type*>(self);
    std::apply([&](const auto&... args) { SetValue(std::move(op->receiver_), args...); },
               op->shared_state_->data_.value());
  }

  friend void tag_invoke(start_t, type& self) {
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

template <typename Sender>
struct _Sender {
  struct type;
};
template <typename Sender>
using sender_t = typename _Sender<remove_cvref_t<Sender>>::type;

template <typename Sender>
struct _Sender<Sender>::type {
  using SharedState = _SharedState<Sender>;
  template <typename Receiver>
  using _operation_t = operation_t<Sender, Receiver>;

  using value_types = completion_signatures_of_t<Sender>;

  Sender sender_;
  std::shared_ptr<SharedState> shared_state_;

  explicit type(Sender sender)
      : sender_(std::move(sender)), shared_state_{std::make_shared<SharedState>(sender_)} {}

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver) -> _operation_t<Receiver> {
    return _operation_t<Receiver>((Receiver &&) receiver, self.shared_state_);
  }
};

struct split_t {
  template <
      typename Sender,
      std::enable_if_t<
          _is_sender<Sender> && _tag_invocable_with_completion_scheduler<split_t, Sender>, int> = 0>
  auto operator()(Sender&& sender) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(split_t{}, std::move(scheduler), (Sender &&) sender);
  }

  template <typename Sender,
            std::enable_if_t<_is_sender<Sender> &&
                                 !_tag_invocable_with_completion_scheduler<split_t, Sender> &&
                                 tag_invocable<split_t, Sender>,
                             int> = 0>
  auto operator()(Sender&& sender) const {
    return tag_invoke(split_t{}, (Sender &&) sender);
  }

  template <typename Sender,
            std::enable_if_t<_is_sender<Sender> &&
                                 !_tag_invocable_with_completion_scheduler<split_t, Sender> &&
                                 !tag_invocable<split_t, Sender>,
                             int> = 0>
  sender_t<Sender> operator()(Sender&& sender) const {
    return sender_t<Sender>{(Sender &&) sender};
  }
  _BinderBack<split_t> operator()() const { return {{}, {}, {}}; }
};

}  // namespace __split

using __split::split_t;
inline constexpr split_t Split{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SPLIT_H_
