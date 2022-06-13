// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXECUTION_TRANSFER_JUST_H_
#define MMDEPLOY_CSRC_EXECUTION_TRANSFER_JUST_H_

#include "mmdeploy/execution/just.h"
#include "mmdeploy/execution/transfer.h"
#include "mmdeploy/execution/utility.h"

namespace mmdeploy {

namespace _transfer_just {

struct transfer_just_t {
  template <typename Scheduler, typename... As,
            std::enable_if_t<tag_invocable<transfer_just_t, Scheduler, As...>, int> = 0>
  auto operator()(Scheduler&& scheduler, As&&... as) const {
    return tag_invoke(transfer_just_t{}, (Scheduler &&) scheduler, (As &&) as...);
  }
  template <typename Scheduler, typename... As,
            std::enable_if_t<!tag_invocable<transfer_just_t, Scheduler, As...>, int> = 0>
  auto operator()(Scheduler&& scheduler, As&&... as) const {
    return Transfer(Just((As &&) as...), (Scheduler &&) scheduler);
  }
};

}  // namespace _transfer_just

using _transfer_just::transfer_just_t;
inline constexpr transfer_just_t TransferJust{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_TRANSFER_JUST_H_
