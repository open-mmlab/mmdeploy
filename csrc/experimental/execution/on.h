// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ON_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ON_H_

#include <variant>

#include "utility.h"

namespace mmdeploy {

namespace __on {

template <typename Scheduler, typename Sender, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Scheduler, typename Sender, typename Receiver>
using operation_t = typename _Operation<Scheduler, Sender, Receiver>::type;

template <typename Scheduler, typename Sender, typename Receiver>
struct _ReceiverRef {
  struct type;
};
template <typename Scheduler, typename Sender, typename Receiver>
using receiver_ref_t = typename _ReceiverRef<Scheduler, Sender, Receiver>::type;

template <typename Scheduler, typename Sender, typename Receiver>
struct _ReceiverRef<Scheduler, Sender, Receiver>::type {
  operation_t<Scheduler, Sender, Receiver>* op_state_;
  template <typename... Args>
  friend void SetValue(type&& self, Args&&... args) {
    SetValue((Receiver &&) self.op_state_->receiver_, ((Args &&) args)...);
  }
};

template <typename Scheduler, typename Sender, typename Receiver>
struct _Receiver {
  struct type;
};
template <typename Scheduler, typename Sender, typename Receiver>
using receiver_t = typename _Receiver<Scheduler, Sender, Receiver>::type;

template <typename Scheduler, typename Sender, typename Receiver>
struct _Receiver<Scheduler, Sender, Receiver>::type {
  operation_t<Scheduler, Sender, Receiver>* op_state_;
  using _receiver_ref_t = receiver_ref_t<Scheduler, Sender, Receiver>;

  friend void SetValue(type&& self) {
    auto op_state = self.op_state_;
    Start(op_state->data_.template emplace<1>(
        Connect((Sender &&) op_state->sender_, _receiver_ref_t{op_state})));
  }
};

template <typename Scheduler, typename Sender, typename Receiver>
struct _Operation<Scheduler, Sender, Receiver>::type {
  using _receiver_t = receiver_t<Scheduler, Sender, Receiver>;
  using _receiver_ref_t = receiver_ref_t<Scheduler, Sender, Receiver>;

  template <class Sender2, class Receiver2>
  type(Scheduler scheduler, Sender2&& sender, Receiver2&& receiver)
      : data_(std::in_place_index<0>, Connect(Schedule(scheduler), _receiver_t{this})),
        scheduler_(scheduler),
        sender_((Sender2 &&) sender),
        receiver_((Receiver2 &&) receiver) {}

  friend void Start(type& self) { Start(std::get<0>(self.data_)); }

  std::variant<connect_result_t<schedule_result_t<Scheduler>, _receiver_t>,
               connect_result_t<Sender, _receiver_ref_t>>
      data_;
  Scheduler scheduler_;
  Sender sender_;
  Receiver receiver_;
};

template <typename Scheduler, typename Predecessor>
struct _Sender {
  struct type;
};
template <typename Scheduler, typename Predecessor>
using sender_t = typename _Sender<std::decay_t<Scheduler>, std::decay_t<Predecessor>>::type;

template <typename Scheduler, typename Predecessor>
struct _Sender<Scheduler, Predecessor>::type {
  using value_type = completion_signature_for_t<Predecessor>;
  Scheduler scheduler_;
  Predecessor pred_;

  template <typename Receiver>
  using _operation_t = operation_t<Scheduler, Predecessor, Receiver>;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver) -> _operation_t<std::decay_t<Receiver>> {
    return {((Self &&) self).scheduler_, ((Self &&) self).pred_, (Receiver &&) receiver};
  }
};

struct on_t {
  template <typename Scheduler, typename Sender>
  sender_t<Scheduler, Sender> operator()(Scheduler&& scheduler, Sender&& sender) const {
    return {(Scheduler &&) scheduler, (Sender &&) sender};
  }
};

}  // namespace __on

using __on::on_t;
inline constexpr on_t On{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ON_H_
