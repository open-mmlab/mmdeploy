// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TRANSFER_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TRANSFER_H_

#include "schedule_from.h"
#include "utility.h"

namespace mmdeploy {

namespace __transfer {

struct transfer_t {
  template <typename Sender, typename Scheduler,
            std::enable_if_t<_is_sender<Sender> && _tag_invocable_with_completion_scheduler<
                                                       transfer_t, Sender, Scheduler>,
                             int> = 0>
  auto operator()(Sender&& sender, Scheduler&& scheduler) const {
    auto sched = GetCompletionScheduler(sender);
    return tag_invoke(transfer_t{}, std::move(sched), (Sender &&) sender, (Scheduler &&) scheduler);
  }
  template <typename Sender, typename Scheduler,
            std::enable_if_t<
                _is_sender<Sender> &&
                    !_tag_invocable_with_completion_scheduler<transfer_t, Sender, Scheduler> &&
                    tag_invocable<transfer_t, Sender, Scheduler>,
                int> = 0>
  auto operator()(Sender&& sender, Scheduler&& scheduler) const {
    return tag_invoke(transfer_t{}, (Sender &&) sender, (Scheduler &&) scheduler);
  }
  template <typename Sender, typename Scheduler,
            std::enable_if_t<
                _is_sender<Sender> &&
                    !_tag_invocable_with_completion_scheduler<transfer_t, Sender, Scheduler> &&
                    !tag_invocable<transfer_t, Sender, Scheduler>,
                int> = 0>
  auto operator()(Sender&& sender, Scheduler&& scheduler) const {
    return ScheduleFrom((Scheduler &&) scheduler, (Sender &&) sender);
  }
  template <typename Scheduler>
  _BinderBack<transfer_t, remove_cvref_t<Scheduler>> operator()(Scheduler&& scheduler) const {
    return {{}, {}, {(Scheduler &&) scheduler}};
  }
};

}  // namespace __transfer

using __transfer::transfer_t;
inline constexpr transfer_t Transfer{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TRANSFER_H_
