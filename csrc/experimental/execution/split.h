// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SPLIT_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SPLIT_H_

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
  friend void SetValue(type&& self, As&&... as) {
    auto& state = self.shared_state_;
    state.data_.emplace((As &&) as...);
    state._Notify();
  }
};

template <typename Predecessor>
struct _SharedState {
  std::optional<completion_signature_for_t<Predecessor>> data_;

  using Receiver = receiver_t<_SharedState>;

  connect_result_t<Predecessor, Receiver> op_state2_;

  std::atomic<void*> head_{nullptr};

  explicit _SharedState(Predecessor& pred)
      : op_state2_(Connect((Predecessor &&) pred, Receiver{*this})) {}

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

template <typename Predecessor, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Predecessor, typename Receiver>
using Operation = typename _Operation<std::decay_t<Predecessor>, std::decay_t<Receiver>>::type;

template <typename Predecessor, typename Receiver>
struct _Operation<Predecessor, Receiver>::type : _OperationBase {
  Receiver receiver_;
  std::shared_ptr<_SharedState<Predecessor>> shared_state_;

  type(Receiver&& receiver, std::shared_ptr<_SharedState<Predecessor>> shared_state)
      : _OperationBase{nullptr, _Notify},
        receiver_(std::move(receiver)),
        shared_state_(std::move(shared_state)) {}

  static void _Notify(_OperationBase* self) noexcept {
    auto op = static_cast<type*>(self);
    std::apply([&](const auto&... args) { SetValue(std::move(op->receiver_), args...); },
               op->shared_state_->data_.value());
  }

  friend void Start(type& self) {
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

template <typename Predecessor>
struct _Sender {
  struct type;
};
template <typename Predecessor>
using Sender = typename _Sender<std::decay_t<Predecessor>>::type;

template <typename Predecessor>
struct _Sender<Predecessor>::type {
  using SharedState = _SharedState<Predecessor>;
  template <typename Receiver>
  using operation_t = Operation<Predecessor, Receiver>;

  using value_type = completion_signature_for_t<Predecessor>;

  Predecessor pred_;
  std::shared_ptr<SharedState> shared_state_;

  explicit type(Predecessor pred)
      : pred_(std::move(pred)), shared_state_{std::make_shared<SharedState>(pred_)} {}

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver) -> operation_t<Receiver> {
    return operation_t<Receiver>((Receiver &&) receiver, self.shared_state_);
  }
};

struct split_t {
  template <typename Predecessor>
  Sender<Predecessor> operator()(Predecessor&& pred) const {
    return Sender<Predecessor>{(Predecessor &&) pred};
  }
  _BinderBack<split_t> operator()() const { return {{}, {}, {}}; }
};

}  // namespace __split

using __split::split_t;
inline constexpr split_t Split{};

}

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SPLIT_H_
