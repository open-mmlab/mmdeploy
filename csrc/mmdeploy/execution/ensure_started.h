// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ENSURE_STARTED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ENSURE_STARTED_H_

#include "concepts.h"
#include "utility.h"

namespace mmdeploy {

namespace __ensure_started {

struct _OperationBase {
  void (*notify_)(_OperationBase*);
};

template <typename SharedState>
struct _Receiver {
  struct type;
};
template <typename SharedState>
using receiver_t = typename _Receiver<SharedState>::type;

template <typename SharedState>
struct _Receiver<SharedState>::type {
  std::shared_ptr<SharedState> shared_state_;

  template <typename... As>
  friend void tag_invoke(set_value_t, type&& self, As&&... as) noexcept {
    assert(self.shared_state_);
    self.shared_state_->data_.emplace((As &&) as...);
    self.shared_state_->_Notify();
    self.shared_state_.reset();
  }
};

template <typename Sender>
struct _SharedState {
  std::optional<completion_signatures_of_t<Sender>> data_;
  //  std::optional<connect_result_t<Sender, receiver_t<_SharedState>>> op_state2_;
  std::optional<__conv_proxy<connect_result_t<Sender, receiver_t<_SharedState>>>> op_state2_proxy_;

  std::atomic<void*> awaiting_{nullptr};

  void _Notify() noexcept {
    void* const completion_state = static_cast<void*>(this);
    void* old = awaiting_.exchange(completion_state, std::memory_order_acq_rel);
    auto* op_state = static_cast<_OperationBase*>(old);

    if (op_state != nullptr) {
      op_state->notify_(op_state);
    }
  }
};

template <typename Sender, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Sender, typename Receiver>
using Operation = typename _Operation<Sender, remove_cvref_t<Receiver>>::type;

template <typename Sender, typename Receiver>
struct _Operation<Sender, Receiver>::type : public _OperationBase {
  Receiver receiver_;
  std::shared_ptr<_SharedState<Sender>> shared_state_;

  type(Receiver&& receiver, std::shared_ptr<_SharedState<Sender>> shared_state)
      : _OperationBase{_Notify},
        receiver_(std::move(receiver)),
        shared_state_(std::move(shared_state)) {}

  static void _Notify(_OperationBase* self) noexcept {
    auto op_state = static_cast<type*>(self);

    std::apply(
        [&](auto&&... vals) -> void {
          SetValue(std::move(op_state->receiver_), (decltype(vals)&&)vals...);
        },
        *op_state->shared_state_->data_);
  }

  friend void tag_invoke(start_t, type& self) {
    auto shared_state = self.shared_state_.get();
    std::atomic<void*>& awaiting = shared_state->awaiting_;
    void* const completion_state = static_cast<void*>(shared_state);
    void* old = awaiting.load(std::memory_order_acquire);

    // TODO: cancel the loop by replacing `compare_exchange_weak` with `compare_exchange_strong`
    do {
      if (old == completion_state) {
        _Notify(&self);
        return;
      }
    } while (awaiting.compare_exchange_weak(old, static_cast<void*>(&self),
                                            std::memory_order_release, std::memory_order_acquire));
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
  using value_types = completion_signatures_of_t<Sender>;

  using SharedState = _SharedState<Sender>;

  std::shared_ptr<SharedState> shared_state_;

  template <typename Sndr, std::enable_if_t<!std::is_same_v<remove_cvref_t<Sndr>, type>, int> = 0>
  explicit type(Sndr&& sender) : shared_state_(std::make_shared<SharedState>()) {
    shared_state_->op_state2_proxy_.emplace(
        [&] { return Connect((Sndr &&) sender, receiver_t<SharedState>{shared_state_}); });
    Start(**shared_state_->op_state2_proxy_);
    //    Start(shared_state_->op_state2_.emplace(
    //        __conv{[&] { return Connect((Sndr &&) sender, receiver_t<SharedState>{shared_state_});
    //        }}));
  }

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver)
      -> Operation<Sender, Receiver> {
    return {(Receiver &&) receiver, std::move(self.shared_state_)};
  }
};

struct ensure_started_t {
  template <typename Sender,
            std::enable_if_t<_is_sender<Sender> &&
                                 _tag_invocable_with_completion_scheduler<ensure_started_t, Sender>,
                             int> = 0>
  auto operator()(Sender&& sender) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(ensure_started_t{}, std::move(scheduler), (Sender &&) sender);
  }

  template <
      typename Sender,
      std::enable_if_t<_is_sender<Sender> &&
                           !_tag_invocable_with_completion_scheduler<ensure_started_t, Sender> &&
                           tag_invocable<ensure_started_t, Sender>,
                       int> = 0>
  auto operator()(Sender&& sender) const {
    return tag_invoke(ensure_started_t{}, (Sender &&) sender);
  }

  template <
      typename Sender,
      std::enable_if_t<_is_sender<Sender> &&
                           !_tag_invocable_with_completion_scheduler<ensure_started_t, Sender> &&
                           !tag_invocable<ensure_started_t, Sender>,
                       int> = 0>
  sender_t<Sender> operator()(Sender&& sender) const {
    return sender_t<Sender>{(Sender &&) sender};
  }
};

}  // namespace __ensure_started

using __ensure_started::ensure_started_t;
inline constexpr ensure_started_t EnsureStarted{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ENSURE_STARTED_H_
