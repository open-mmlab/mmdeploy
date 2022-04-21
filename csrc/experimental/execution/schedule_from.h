// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SCHEDULE_FROM_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SCHEDULE_FROM_H_

#include "utility.h"
#include <optional>

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

  friend void SetValue(type&& self) noexcept {
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
  friend void SetValue(type&& self, As&&... as) {
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
  std::optional<completion_signature_for_t<std::decay_t<CvrefSender>>> data_;
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

  friend void Start(type& op_state) noexcept { Start(op_state.op_state1_); }
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
  Predecessor sender_;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver)
      -> operation1_t<Scheduler, _copy_cvref_t<Self, Predecessor>, std::decay_t<Receiver>> {
    return {self.scheduler_, ((Self &&) self).sender_, (Receiver &&) receiver};
  }

  friend Scheduler GetCompletionScheduler(const type& self) noexcept { return self.scheduler_; }
};

struct schedule_from_t {
  template <typename Scheduler, typename Predecessor>
  sender_t<Scheduler, Predecessor> operator()(Scheduler&& scheduler, Predecessor&& pred) const {
    return {(Scheduler &&) scheduler, (Predecessor &&) pred};
  }
};

}  // namespace __schedule_from

using __schedule_from::schedule_from_t;
inline constexpr schedule_from_t ScheduleFrom{};

}

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SCHEDULE_FROM_H_
