// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_EXECUTE_H_
#define MMDEPLOY_CSRC_EXECUTION_EXECUTE_H_

#include "execution/start_detached.h"
#include "execution/then.h"
#include "execution/utility.h"

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
