// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_DYNAMIC_BATCH_H_
#define MMDEPLOY_CSRC_EXECUTION_DYNAMIC_BATCH_H_

#include <atomic>

#include "mmdeploy/execution/then.h"
#include "mmdeploy/execution/utility.h"

namespace mmdeploy {

namespace _dynamic_batch {

struct dynamic_batch_t {
  struct context_base_t {
    void (*destroy_)(context_base_t*);
  };
  struct context_t {
    std::atomic<context_base_t*> base{};
    ~context_t() {
      if (auto p = base.load()) {
        p->destroy_(p);
      }
    }
  };

  template <typename Sender, typename Func,
            std::enable_if_t<
                _tag_invocable_with_completion_scheduler<dynamic_batch_t, Sender, context_t&, Func>,
                int> = 0>
  auto operator()(Sender&& sender, context_t& context, Func func) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(*this, std::move(scheduler), (Sender &&) sender, context, std::move(func));
  }

  template <typename Sender, typename Func,
            std::enable_if_t<!_tag_invocable_with_completion_scheduler<dynamic_batch_t, Sender,
                                                                       context_t&, Func> &&
                                 tag_invocable<dynamic_batch_t, Sender, context_t&, Func>,
                             int> = 0>
  auto operator()(Sender&& sender, context_t& context, Func func) const {
    return tag_invoke(*this, (Sender &&) sender, context, std::move(func));
  }

  template <typename Sender, typename Context, typename Func,
            std::enable_if_t<
                !_tag_invocable_with_completion_scheduler<dynamic_batch_t, Sender, Context, Func> &&
                    !tag_invocable<dynamic_batch_t, Sender, Context, Func>,
                int> = 0>
  auto operator()(Sender&& sender, Context&&, Func func) const {
    return Then((Sender &&) sender, std::move(func));
  }
};

}  // namespace _dynamic_batch

using _dynamic_batch::dynamic_batch_t;
inline constexpr dynamic_batch_t DynamicBatch{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_DYNAMIC_BATCH_H_
