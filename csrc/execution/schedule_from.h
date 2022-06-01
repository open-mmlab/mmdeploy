// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SCHEDULE_FROM_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SCHEDULE_FROM_H_

#include <optional>

#include "utility.h"

namespace mmdeploy {

namespace __schedule_from {

template <typename Scheduler, typename CvrefSender, typename Receiver>
struct _Operation1 {
  struct type;
};
template <typename Scheduler, typename CvrefSender, typename Receiver>
using operation1_t = typename _Operation1<Scheduler, CvrefSender, Receiver>::type;

template <typename Scheduler, typename CvrefSender, typename Receiver>
struct _Receiver1 {
  struct type;
};
template <typename Scheduler, typename CvrefSender, typename Receiver>
using receiver1_t = typename _Receiver1<Scheduler, CvrefSender, Receiver>::type;

template <typename Scheduler, typename CvrefSender, typename Receiver>
struct _Receiver2 {
  struct type;
};
template <typename Scheduler, typename CvrefSender, typename Receiver>
using receiver2_t = typename _Receiver2<Scheduler, CvrefSender, Receiver>::type;

template <typename Scheduler, typename CvrefSender, typename Receiver>
struct _Receiver2<Scheduler, CvrefSender, Receiver>::type {
  operation1_t<Scheduler, CvrefSender, Receiver>* op_state_;

  friend void tag_invoke(set_value_t, type&& self) noexcept {
    std::apply(
        [&](auto&&... vals) {
          SetValue(std::move(self.op_state_->receiver_), std::move(vals)...);  //
        },
        std::move(*self.op_state_->data_));
  }
};

template <typename Scheduler, typename CvrefSender, typename Receiver>
struct _Receiver1<Scheduler, CvrefSender, Receiver>::type {
  using _receiver2_t = receiver2_t<Scheduler, CvrefSender, Receiver>;

  operation1_t<Scheduler, CvrefSender, Receiver>* op_state_;

  template <typename... As>
  friend void tag_invoke(set_value_t, type&& self, As&&... as) noexcept {
    self.op_state_->data_.emplace((As &&) as...);
    auto sender = Schedule(self.op_state_->scheduler_);
    auto& op_state2 = self.op_state_->op_state2_.emplace(
        __conv{[&] { return Connect(std::move(sender), _receiver2_t{self.op_state_}); }});
    Start(op_state2);
  }
};

template <typename Scheduler, typename CvrefSender, typename Receiver>
struct _Operation1<Scheduler, CvrefSender, Receiver>::type {
  using _receiver1_t = receiver1_t<Scheduler, CvrefSender, Receiver>;
  using _receiver2_t = receiver2_t<Scheduler, CvrefSender, Receiver>;

  Scheduler scheduler_;
  Receiver receiver_;
  std::optional<completion_signatures_of_t<remove_cvref_t<CvrefSender>>> data_;
  connect_result_t<CvrefSender, _receiver1_t> op_state1_;
  std::optional<connect_result_t<schedule_result_t<Scheduler>, _receiver2_t>> op_state2_;

  template <class Receiver2>
  type(Scheduler sched, CvrefSender&& sender, Receiver2&& receiver)
      : scheduler_(sched),
        receiver_((Receiver2 &&) receiver),
        op_state1_(Connect((CvrefSender &&) sender, _receiver1_t{this})) {}

  type(const type&) = delete;
  type(_Operation1&&) noexcept = delete;
  type& operator=(const type&) = delete;
  type& operator=(type&&) noexcept = delete;

  friend void tag_invoke(start_t, type& op_state) noexcept { Start(op_state.op_state1_); }
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

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver)
      -> operation1_t<Scheduler, _copy_cvref_t<Self, Sender>, remove_cvref_t<Receiver>> {
    return {self.scheduler_, ((Self &&) self).sender_, (Receiver &&) receiver};
  }

  friend Scheduler tag_invoke(get_completion_scheduler_t, const type& self) noexcept {
    return self.scheduler_;
  }
};

struct schedule_from_t {
  template <typename Scheduler, typename Sender,
            std::enable_if_t<
                _is_sender<Sender> && tag_invocable<schedule_from_t, Scheduler, Sender>, int> = 0>
  auto operator()(Scheduler&& scheduler, Sender&& sender) const
      -> tag_invoke_result_t<schedule_from_t, Scheduler, Sender> {
    return tag_invoke(schedule_from_t{}, (Scheduler &&) scheduler, (Sender &&) sender);
  }

  template <typename Scheduler, typename Sender,
            std::enable_if_t<
                _is_sender<Sender> && !tag_invocable<schedule_from_t, Scheduler, Sender>, int> = 0>
  sender_t<Scheduler, Sender> operator()(Scheduler&& scheduler, Sender&& sender) const {
    return {(Scheduler &&) scheduler, (Sender &&) sender};
  }
};

}  // namespace __schedule_from

using __schedule_from::schedule_from_t;
inline constexpr schedule_from_t ScheduleFrom{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SCHEDULE_FROM_H_
