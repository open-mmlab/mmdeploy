// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_

#include "execution.h"

namespace mmdeploy {

struct InlineScheduler {
  template <typename Receiver>
  struct _Operation {
    Receiver receiver_;
    friend void Start(_Operation& op) noexcept { SetValue((Receiver &&) op.receiver_); }
  };

  struct _Sender {
    using value_types = std::tuple<>;

    template <typename Receiver>
    friend auto Connect(_Sender, Receiver&& receiver) -> _Operation<std::decay_t<Receiver>> {
      return {(Receiver &&) receiver};
    }

    friend InlineScheduler tag_invoke(get_completion_scheduler_t, const _Sender&) { return {}; }
  };

  _Sender Schedule() const noexcept { return {}; }

  friend void* GetSchedulerId(const InlineScheduler& self) { return (void*)1u; }

  template <typename Sender>
  struct _Receiver {
    std::optional<completion_signatures_of_t<Sender>>* data_;
    template <typename... As>
    friend void SetValue(_Receiver&& r, As&&... as) noexcept {
      r.data_->emplace((As &&) as...);
    }
  };

  //  template <class S, class Sender = std::decay_t<S>,
  //            class Tuple = completion_signature_for_t<Sender>>
  //  friend Tuple SyncWait(InlineScheduler, S&& sender) {
  //    std::optional<Tuple> data;
  //    auto op_state = Connect(((S &&) sender), _Receiver<Sender>{&data});
  //    Start(op_state);
  //    return std::move(data).value();
  //  }
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_
