// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TRANSFER_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TRANSFER_H_

#include "schedule_from.h"
#include "utility.h"

namespace mmdeploy {

namespace __transfer {

struct transfer_t {
  template <typename Sender, typename Scheduler>
  auto operator()(Sender&& sender, Scheduler&& scheduler) const {
    return ScheduleFrom((Scheduler &&) scheduler, (Sender &&) sender);
  }

  template <typename Scheduler>
  _BinderBack<transfer_t, std::decay_t<Scheduler>> operator()(Scheduler&& scheduler) const {
    return {{}, {}, {(Scheduler &&) scheduler}};
  }
};

}  // namespace __transfer

using __transfer::transfer_t;
inline constexpr transfer_t Transfer{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TRANSFER_H_
