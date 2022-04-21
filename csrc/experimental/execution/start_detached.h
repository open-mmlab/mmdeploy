// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_START_DETACHED_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_START_DETACHED_H_

#include "submit.h"
#include "utility.h"

namespace mmdeploy {

namespace __start_detached {

struct _Receiver {
  template <typename... As>
  friend void SetValue(_Receiver&&, As&&...) noexcept {}
};

struct start_detached_t {
  template <typename Predecessor>
  void operator()(Predecessor&& pred) const {
    __Submit((Predecessor &&) pred, _Receiver{});
  }
};

}  // namespace __start_detached

using __start_detached::start_detached_t;
inline constexpr start_detached_t StartDetached{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_START_DETACHED_H_
