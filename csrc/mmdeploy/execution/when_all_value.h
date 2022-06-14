// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_VALUE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_VALUE_H_

#include "mmdeploy/core/value.h"
#include "mmdeploy/execution/schedulers/registry.h"

namespace mmdeploy {

namespace __when_all_value {

template <typename Receiver>
struct _Operation {
  struct type;
};
template <typename Receiver>
using operation_t = typename _Operation<Receiver>::type;

template <typename Receiver>
struct _Receiver {
  struct type;
};
template <typename Receiver>
using receiver_t = typename _Receiver<Receiver>::type;

template <typename Receiver>
struct _Receiver<Receiver>::type {
  size_t index_;
  operation_t<Receiver>* op_state_;

  friend void tag_invoke(set_value_t, type&& self, Value value) noexcept {
    self.op_state_->values_[self.index_] = std::move(value);
    if (0 == --self.op_state_->count_) {
      SetValue(std::move(self.op_state_->rcvr_), std::move(self.op_state_->values_));
    }
  }
};

template <typename Receiver>
struct _Operation<Receiver>::type {
  std::vector<TypeErasedOperation> ConnectChildren(std::vector<TypeErasedSender<Value>> senders) {
    std::vector<TypeErasedOperation> op_states;
    op_states.reserve(senders.size());
    for (size_t i = 0; i < senders.size(); ++i)
      op_states.push_back(Connect(std::move(senders[i]), receiver_t<Receiver>{i, this}));
    return op_states;
  }
  type(std::vector<TypeErasedSender<Value>> senders, Receiver receiver)
      : child_op_states_{ConnectChildren(std::move(senders))},
        rcvr_((Receiver &&) receiver),
        count_(child_op_states_.size()),
        values_(child_op_states_.size()) {}

  std::vector<TypeErasedOperation> child_op_states_;
  Receiver rcvr_;
  std::atomic<size_t> count_;
  std::vector<Value> values_;

  friend void tag_invoke(start_t, type& op_state) {
    for (auto& op : op_state.child_op_states_) {
      Start(op);
    }
  }
};

struct sender_t {
  using value_types = std::tuple<std::vector<Value>>;

  std::vector<TypeErasedSender<Value>> senders_;

  template <typename Self, typename Receiver, typename = _decays_to<Self, sender_t>>
  friend operation_t<remove_cvref_t<Receiver>> tag_invoke(connect_t, Self&& self,
                                                          Receiver&& receiver) {
    return {((Self &&) self).senders_, (Receiver &&) receiver};
  }
};

}  // namespace __when_all_value

namespace _type_erased {

inline __when_all_value::sender_t tag_invoke(when_all_t,
                                             std::vector<TypeErasedSender<Value>> senders) {
  return {std::move(senders)};
}

}  // namespace _type_erased

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_WHEN_ALL_VALUE_H_
