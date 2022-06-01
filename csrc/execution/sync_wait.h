// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SYNC_WAIT_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SYNC_WAIT_H_

#include <optional>

#include "run_loop.h"
#include "utility.h"

namespace mmdeploy {

namespace __sync_wait {

template <typename Sender>
struct _State {
  std::optional<completion_signatures_of_t<Sender>> data_;
};

template <typename Sender>
struct _Receiver {
  struct type;
};
template <typename Sender>
using receiver_t = typename _Receiver<remove_cvref_t<Sender>>::type;

template <typename Sender>
struct _Receiver<Sender>::type {
  _State<Sender>* state_;
  RunLoop* loop_;

  template <typename... As>
  friend void tag_invoke(set_value_t, type&& receiver, As&&... as) noexcept {
    receiver.state_->data_.emplace((As &&) as...);
    receiver.loop_->_Finish();
  }
};

struct sync_wait_t {
  template <typename Sender,
            std::enable_if_t<_is_sender<Sender> &&
                                 _tag_invocable_with_completion_scheduler<sync_wait_t, Sender>,
                             int> = 0>
  auto operator()(Sender&& sender) const
      -> tag_invoke_result_t<sync_wait_t, _completion_scheduler_for<Sender>, Sender> {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(sync_wait_t{}, std::move(scheduler), (Sender &&) sender);
  }
  template <typename Sender,
            std::enable_if_t<_is_sender<Sender> &&
                                 !_tag_invocable_with_completion_scheduler<sync_wait_t, Sender> &&
                                 tag_invocable<sync_wait_t, Sender>,
                             int> = 0>
  auto operator()(Sender&& sender) const -> tag_invoke_result_t<sync_wait_t, Sender> {
    return tag_invoke(sync_wait_t{}, (Sender &&) sender);
  }

  template <typename Sender,
            std::enable_if_t<_is_sender<Sender> &&
                                 !_tag_invocable_with_completion_scheduler<sync_wait_t, Sender> &&
                                 !tag_invocable<sync_wait_t, Sender>,
                             int> = 0>
  completion_signatures_of_t<Sender> operator()(Sender&& sender) const {
    _State<remove_cvref_t<Sender>> state;
    RunLoop loop;
    // connect to internal receiver
    auto op_state = Connect((Sender &&) sender, receiver_t<Sender>{&state, &loop});
    Start(op_state);

    loop._Run();
    // extract the returned values
    return std::move(*state.data_);
  }
};

}  // namespace __sync_wait

using __sync_wait::sync_wait_t;
inline constexpr sync_wait_t SyncWait{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SYNC_WAIT_H_
