// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

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
  friend void tag_invoke(set_value_t, type&& self, Args&&... args) noexcept {
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

  friend void tag_invoke(set_value_t, type&& self) noexcept {
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

  friend void tag_invoke(start_t, type& self) { Start(std::get<0>(self.data_)); }

  std::variant<connect_result_t<schedule_result_t<Scheduler>, _receiver_t>,
               connect_result_t<Sender, _receiver_ref_t>>
      data_;
  Scheduler scheduler_;
  Sender sender_;
  Receiver receiver_;
};

template <typename Scheduler, typename Sender>
struct _Sender {
  struct type;
};
template <typename Scheduler, typename Sender>
using sender_t = typename _Sender<remove_cvref_t<Scheduler>, remove_cvref_t<Sender>>::type;

template <typename Scheduler, typename Sender>
struct _Sender<Scheduler, Sender>::type {
  using value_types = completion_signatures_of_t<Sender>;
  Scheduler scheduler_;
  Sender sender_;

  template <typename Receiver>
  using _operation_t = operation_t<Scheduler, Sender, remove_cvref_t<Receiver>>;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver) -> _operation_t<Receiver> {
    return {((Self &&) self).scheduler_, ((Self &&) self).sender_, (Receiver &&) receiver};
  }
};

struct on_t {
  template <typename Scheduler, typename Sender,
            std::enable_if_t<_is_sender<Sender> && tag_invocable<on_t, Scheduler, Sender>, int> = 0>
  auto operator()(Scheduler&& scheduler, Sender&& sender) const
      -> tag_invoke_result_t<on_t, Scheduler, Sender> {
    return tag_invoke(on_t{}, (Scheduler &&) scheduler, (Sender &&) sender);
  }
  template <
      typename Scheduler, typename Sender,
      std::enable_if_t<_is_sender<Sender> && !tag_invocable<on_t, Scheduler, Sender>, int> = 0>
  sender_t<Scheduler, Sender> operator()(Scheduler&& scheduler, Sender&& sender) const {
    return {(Scheduler &&) scheduler, (Sender &&) sender};
  }
};

}  // namespace __on

using __on::on_t;
inline constexpr on_t On{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_ON_H_
