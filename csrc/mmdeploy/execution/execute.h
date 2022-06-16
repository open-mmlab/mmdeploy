// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXECUTION_EXECUTE_H_
#define MMDEPLOY_CSRC_EXECUTION_EXECUTE_H_

#include "mmdeploy/execution/start_detached.h"
#include "mmdeploy/execution/then.h"
#include "mmdeploy/execution/utility.h"

namespace mmdeploy {

namespace _execute {

struct execute_t {
  template <typename Scheduler, typename Func,
            std::enable_if_t<tag_invocable<execute_t, Scheduler, Func>, int> = 0>
  void operator()(Scheduler&& scheduler, Func func) const {
    return tag_invoke(*this, (Scheduler &&) scheduler, std::move(func));
  }
  template <typename Scheduler, typename Func,
            std::enable_if_t<!tag_invocable<execute_t, Scheduler, Func>, int> = 0>
  void operator()(Scheduler&& scheduler, Func func) const {
    return StartDetached(Then(Schedule((Scheduler &&) scheduler), std::move(func)));
  }
};

}  // namespace _execute

using _execute::execute_t;
inline constexpr execute_t Execute{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_EXECUTE_H_
