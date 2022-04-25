// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_

#include "execution/execution.h"

namespace mmdeploy {

namespace _inline_sched {

template <typename Receiver>
struct _Operation {
  struct type;
};
template <typename Receiver>
using operation_t = typename _Operation<remove_cvref_t<Receiver>>::type;

template <typename Receiver>
struct _Operation<Receiver>::type {
  Receiver receiver_;
  friend void Start(type& op) noexcept { SetValue(std::move(op.receiver_)); }
};

struct _Sender {
  using value_types = std::tuple<>;

  template <typename Receiver>
  friend auto Connect(_Sender, Receiver&& receiver) -> operation_t<Receiver> {
    return {(Receiver &&) receiver};
  }

  //
};

struct InlineScheduler {
  _inline_sched::_Sender Schedule() const noexcept { return {}; }
  friend void* GetSchedulerId(const InlineScheduler& self) { return (void*)1u; }
};

inline InlineScheduler tag_invoke(get_completion_scheduler_t, const _Sender&) { return {}; }

template <typename Sender>
struct _Receiver {
  struct type;
};
template <typename Sender>
using receiver_t = typename _Receiver<remove_cvref_t<Sender>>::type;

template <typename Sender>
struct _Receiver<Sender>::type {
  std::optional<completion_signatures_of_t<Sender>>* data_;
  template <typename... As>
  friend void SetValue(type&& r, As&&... as) noexcept {
    r.data_->emplace((As &&) as...);
  }
};

template <typename Sender>
completion_signatures_of_t<Sender> tag_invoke(sync_wait_t, InlineScheduler, Sender&& sender) {
  std::optional<completion_signatures_of_t<Sender>> data;
  auto op_state = Connect(((Sender &&) sender), _inline_sched::receiver_t<Sender>{&data});
  Start(op_state);
  return std::move(data).value();
}

}  // namespace _inline_sched

using _inline_sched::InlineScheduler;
inline constexpr InlineScheduler inline_scheduler{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INLINED_SCHEDULER_H_
