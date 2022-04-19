// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_

#include "execution.h"

namespace mmdeploy {

struct InlineScheduler {
  template <typename R>
  struct _Operation {
    R rec_;
    friend void Start(_Operation& op) noexcept { SetValue((R &&) op.rec_); }
  };

  struct _Sender {
    using value_type = std::tuple<>;

    template <typename R>
    friend auto Connect(_Sender, R&& rec) -> _Operation<std::decay_t<R>> {
      return {(R &&) rec};
    }

    friend InlineScheduler GetCompletionScheduler(_Sender) noexcept { return {}; }
  };

  friend _Sender Schedule(const InlineScheduler) noexcept { return {}; }

  friend void* GetSchedulerId(const InlineScheduler& self) { return (void*)1u; }

  template <class Sender>
  struct _Receiver {
    std::optional<completion_signature_for_t<Sender>>* data_;
    template <class... As>
    friend void SetValue(_Receiver&& r, As&&... as) noexcept {
      r.data_->emplace((As &&) as...);
    }
  };

  template <class S, class Sender = std::decay_t<S>,
            class Tuple = completion_signature_for_t<Sender>>
  friend Tuple SyncWait(InlineScheduler, S&& sender) {
    std::optional<Tuple> data;
    auto op_state = Connect(((S &&) sender), _Receiver<Sender>{&data});
    Start(op_state);
    return std::move(data).value();
  }
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_
