// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_JUST_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_JUST_H_

#include <tuple>

#include "concepts.h"
#include "utility.h"

namespace mmdeploy {

namespace __just {

template <typename Receiver, typename... Ts>
struct _Operation {
  struct type;
};
template <typename Receiver, typename... Ts>
using operation_t = typename _Operation<remove_cvref_t<Receiver>, Ts...>::type;

template <typename Receiver, typename... Ts>
struct _Operation<Receiver, Ts...>::type {
  std::tuple<Ts...> values_;
  Receiver receiver_;
  friend void tag_invoke(start_t, type& op_state) noexcept {
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
using sender_t = typename _Sender<std::decay_t<Ts>...>::type;

template <typename... Ts>
struct _Sender<Ts...>::type {
  using value_types = std::tuple<Ts...>;
  value_types values_;

  template <typename Receiver>
  friend operation_t<Receiver, Ts...> tag_invoke(connect_t, const type& self, Receiver&& receiver) {
    return {self.values_, (Receiver &&) receiver};
  }

  template <typename Receiver>
  friend operation_t<Receiver, Ts...> tag_invoke(connect_t, type&& self, Receiver&& receiver) {
    return {std::move(self).values_, (Receiver &&) receiver};
  }
};

struct just_t {
  template <typename... Ts>
  sender_t<Ts...> operator()(Ts&&... ts) const {
    return {{(Ts &&) ts...}};
  }
};

}  // namespace __just

using __just::just_t;
inline constexpr just_t Just{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_JUST_H_
