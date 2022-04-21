// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ENSURE_STARTED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ENSURE_STARTED_H_

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
  friend void SetValue(type&& self, As&&... as) {
    assert(self.shared_state_);
    self.shared_state_->data_.emplace((As &&) as...);
    self.shared_state_->_Notify();
    self.shared_state_.reset();
  }
};

template <typename Predecessor>
struct _SharedState {
  std::optional<completion_signature_for_t<Predecessor>> data_;
  std::optional<connect_result_t<Predecessor, receiver_t<_SharedState>>> op_state2_;
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

template <typename Predecessor, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Predecessor, typename Receiver>
using Operation = typename _Operation<Predecessor, std::decay_t<Receiver>>::type;

template <typename Predecessor, typename Receiver>
struct _Operation<Predecessor, Receiver>::type : public _OperationBase {
  Receiver receiver_;
  std::shared_ptr<_SharedState<Predecessor>> shared_state_;

  type(Receiver&& receiver, std::shared_ptr<_SharedState<Predecessor>> shared_state)
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

  friend void Start(type& self) {
    auto shared_state = self.shared_state_.get();
    std::atomic<void*>& awaiting = shared_state->awaiting_;
    void* const completion_state = static_cast<void*>(shared_state);
    void* old = awaiting.load(std::memory_order_acquire);

    do {
      if (old == completion_state) {
        _Notify(&self);
        return;
      }
    } while (awaiting.compare_exchange_weak(old, static_cast<void*>(&self),
                                            std::memory_order_release, std::memory_order_acquire));
  }
};

template <typename Predecessor>
struct _Sender {
  struct type;
};
template <typename Predecessor>
using Sender = typename _Sender<std::decay_t<Predecessor>>::type;

template <typename Predecessor>
struct _Sender<Predecessor>::type {
  using SharedState = _SharedState<Predecessor>;
  using value_type = completion_signature_for_t<Predecessor>;

  std::shared_ptr<SharedState> shared_state_;

  template <typename Pred, std::enable_if_t<!std::is_same_v<std::decay_t<Pred>, type>, int> = 0>
  explicit type(Pred&& pred) : shared_state_(std::make_shared<SharedState>()) {
    Start(shared_state_->op_state2_.emplace(
        __conv{[&] { return Connect((Pred &&) pred, receiver_t<SharedState>{shared_state_}); }}));
  }

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver) -> Operation<Predecessor, Receiver> {
    return {(Receiver &&) receiver, std::move(self.shared_state_)};
  }
};

struct ensure_started_t {
  template <typename Predecessor, std::enable_if_t<_decays_to_sender<Predecessor>, int> = 0>
  Sender<Predecessor> operator()(Predecessor&& pred) const {
    return Sender<Predecessor>{(Predecessor &&) pred};
  }
};

}  // namespace __ensure_started

using __ensure_started::ensure_started_t;
inline constexpr ensure_started_t EnsureStarted{};

}

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ENSURE_STARTED_H_
