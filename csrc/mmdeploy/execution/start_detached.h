// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_START_DETACHED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_START_DETACHED_H_

#include "submit.h"
#include "utility.h"

namespace mmdeploy {

namespace __start_detached {

struct _Receiver {
  template <typename... As>
  friend void tag_invoke(set_value_t, _Receiver&&, As&&...) noexcept {}
};

struct start_detached_t {
  template <
      typename Sender,
      std::enable_if_t<_is_sender<Sender> && tag_invocable<start_detached_t, Sender>, int> = 0>
  void operator()(Sender&& sender) const {
    (void)tag_invoke(start_detached_t{}, (Sender &&) sender);
  }

  template <
      typename Sender,
      std::enable_if_t<_is_sender<Sender> && !tag_invocable<start_detached_t, Sender>, int> = 0>
  void operator()(Sender&& sender) const {
    __Submit((Sender &&) sender, _Receiver{});
  }
};

}  // namespace __start_detached

using __start_detached::start_detached_t;
inline constexpr start_detached_t StartDetached{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_START_DETACHED_H_
