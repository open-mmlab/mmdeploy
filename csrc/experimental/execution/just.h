// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_JUST_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_JUST_H_

#include <tuple>

#include "utility.h"

namespace mmdeploy {

namespace __just {

template <typename Receiver, typename... Ts>
struct _Operation {
  struct type;
};
template <typename Receiver, typename... Ts>
using Operation = typename _Operation<std::decay_t<Receiver>, Ts...>::type;

template <typename Receiver, typename... Ts>
struct _Operation<Receiver, Ts...>::type {
  std::tuple<Ts...> values_;
  Receiver receiver_;
  friend void Start(type& op_state) noexcept {
    std::apply(
        [&](Ts&... ts) -> void { SetValue(std::move(op_state.receiver_), std::move(ts)...); },
        op_state.values_);
  }
};

template <typename... Ts>
struct _Sender {
  struct type;
};
template <typename... Ts>
using Sender = typename _Sender<std::decay_t<Ts>...>::type;

template <typename... Ts>
struct _Sender<Ts...>::type {
  using value_type = std::tuple<Ts...>;
  value_type values_;

  template <typename Receiver>
  friend Operation<Receiver, Ts...> Connect(const type& self, Receiver&& receiver) {
    return {self.values_, (Receiver &&) receiver};
  }

  template <typename Receiver>
  friend Operation<Receiver, Ts...> Connect(type&& self, Receiver&& receiver) {
    return {std::move(self).values_, (Receiver &&) receiver};
  }
};

struct just_t {
  template <typename... Ts>
  Sender<Ts...> operator()(Ts&&... ts) const {
    return Sender<Ts...>{{(Ts &&) ts...}};
  }
};

}  // namespace __just

using __just::just_t;
inline constexpr just_t Just{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_JUST_H_
