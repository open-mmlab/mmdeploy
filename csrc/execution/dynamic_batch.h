// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_DYNAMIC_BATCH_H_
#define MMDEPLOY_CSRC_EXECUTION_DYNAMIC_BATCH_H_

#include <atomic>

#include "execution/then.h"
#include "execution/utility.h"

namespace mmdeploy {

namespace _dynamic_batch {

struct dynamic_batch_t {
  struct context_base_t {
    void (*destroy_)(context_base_t*);
  };

  template <typename Sender, typename Manager, typename Func,
            std::enable_if_t<
                _tag_invocable_with_completion_scheduler<dynamic_batch_t, Sender, Manager&, Func>,
                int> = 0>
  auto operator()(Sender&& sender, Manager& manager, Func func) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(*this, std::move(scheduler), (Sender &&) sender, manager, std::move(func));
  }

  template <typename Sender, typename Manager, typename Func,
            std::enable_if_t<
                !_tag_invocable_with_completion_scheduler<dynamic_batch_t, Sender, Manager&, Func>,
                int> = 0>
  auto operator()(Sender&& sender, Manager&&, Func func) const {
    return Then((Sender &&) sender, std::move(func));
  }
};

}  // namespace _dynamic_batch

using _dynamic_batch::dynamic_batch_t;
inline constexpr dynamic_batch_t DynamicBatch{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_DYNAMIC_BATCH_H_
