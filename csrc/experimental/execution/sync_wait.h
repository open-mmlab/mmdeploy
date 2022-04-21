// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SYNC_WAIT_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SYNC_WAIT_H_

#include <optional>

#include "run_loop.h"
#include "utility.h"

namespace mmdeploy {

namespace __sync_wait {

template <typename Sender>
struct _State {
  std::optional<completion_signature_for_t<Sender>> data_;
};

template <typename Sender>
struct _Receiver {
  struct type;
};
template <typename Sender>
using receiver_t = typename _Receiver<std::decay_t<Sender>>::type;

template <typename Sender>
struct _Receiver<Sender>::type {
  _State<Sender>* state_;
  RunLoop* loop_;

  template <typename... As>
  friend void SetValue(type&& receiver, As&&... as) noexcept {
    receiver.state_->data_.emplace((As &&) as...);
    receiver.loop_->_Finish();
  }
};

struct sync_wait_t {
  template <typename Sender>
  completion_signature_for_t<std::decay_t<Sender>> operator()(Sender&& sender) const {
    _State<std::decay_t<Sender>> state;
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
