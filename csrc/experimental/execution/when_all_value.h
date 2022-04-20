// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_VALUE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_VALUE_H_

#include "core/value.h"
#include "execution.h"

namespace mmdeploy {

namespace __when_all_value {

template <class Receiver>
struct __Operation {
  struct type;
};

template <class Receiver>
using _Operation = typename __Operation<Receiver>::type;

template <class Receiver>
struct __Receiver {
  struct type;
};

template <class Receiver>
using _Receiver = typename __Receiver<Receiver>::type;

template <class Receiver>
struct __Receiver<Receiver>::type {
  size_t index_;
  _Operation<Receiver>* op_state_;

  friend void SetValue(type&& self, Value val) noexcept {
    self.op_state_->values_[self.index_] = std::move(val);
    if (0 == --self.op_state_->count_) {
      SetValue(std::move(self.op_state_->rcvr_), std::move(self.op_state_->values_));
    }
  }
};

template <class Receiver>
struct __Operation<Receiver>::type {
  std::vector<TypeErasedOperation<Value>> ConnectChildren(
      std::vector<TypeErasedSender<Value>> sndrs) {
    std::vector<TypeErasedOperation<Value>> op_states;
    op_states.reserve(sndrs.size());
    for (size_t i = 0; i < sndrs.size(); ++i)
      op_states.push_back(Connect(std::move(sndrs[i]), _Receiver<Receiver>{i, this}));
    return op_states;
  }

  type(std::vector<TypeErasedSender<Value>> sndrs, Receiver rcvr)
      : child_op_states_{ConnectChildren(std::move(sndrs))},
        rcvr_((Receiver &&) rcvr),
        count_(child_op_states_.size()),
        values_(child_op_states_.size()) {}

  std::vector<TypeErasedOperation<Value>> child_op_states_;
  Receiver rcvr_;
  std::atomic<size_t> count_;
  std::vector<Value> values_;

  friend void Start(type& op_state) {
    for (auto& op : op_state.child_op_states_) {
      Start(op);
    }
  }
};

struct __Sender {
  struct type;
};

using _Sender = __Sender::type;

struct __Sender::type {
  using value_type = std::tuple<std::vector<Value>>;

  std::vector<TypeErasedSender<Value>> sndrs_;

  template <class Self, class Receiver, class = _decays_to<Self, type>>
  friend _Operation<std::decay_t<Receiver>> Connect(Self&& self, Receiver&& rcvr) {
    return {((Self &&) self).sndrs_, (Receiver &&) rcvr};
  }
};

}  // namespace __when_all_value

inline __when_all_value::_Sender WhenAll_(std::vector<TypeErasedSender<Value>> sndrs) {
  return {std::move(sndrs)};
}

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_VALUE_H_
