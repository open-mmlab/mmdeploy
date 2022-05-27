// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_THEN_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_THEN_H_

#include "closure.h"
#include "concepts.h"
#include "utility.h"

namespace mmdeploy {

namespace __then {

template <typename Receiver, typename Func>
struct _Receiver {
  struct type;
};
template <typename Receiver, typename Func>
using receiver_t = typename _Receiver<Receiver, Func>::type;

template <typename Receiver, typename Func>
struct _Receiver<Receiver, Func>::type {
  Receiver receiver_;
  Func func_;

  template <typename... Args>
  friend void tag_invoke(set_value_t, type&& self, Args&&... args) noexcept {
    if constexpr (std::is_void_v<std::invoke_result_t<Func&&, Args...>>) {
      std::invoke(std::move(self.func_), (Args &&) args...);
      SetValue(std::move(self.receiver_));
    } else {
      SetValue(std::move(self.receiver_), std::invoke(std::move(self.func_), (Args &&) args...));
    }
  }
};

template <typename Sender, typename Func>
struct _Sender {
  struct type;
};
template <typename Sender, typename Func>
using sender_t = typename _Sender<remove_cvref_t<Sender>, remove_cvref_t<Func>>::type;

template <typename Sender, typename Func>
struct _Sender<Sender, Func>::type {
  using _ret_type = decltype(
      std::apply(std::declval<Func>(), std::declval<completion_signatures_of_t<Sender>>()));

  using value_types =
      std::conditional_t<std::is_void_v<_ret_type>, std::tuple<>, std::tuple<_ret_type>>;

  Sender sender_;
  Func func_;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
    return Connect(((Self &&) self).sender_,
                   receiver_t<Receiver, Func>{(Receiver &&) receiver, std::move(self.func_)});
  }

  template <typename SenderT = Sender>
  friend auto tag_invoke(get_completion_scheduler_t, const type& self) noexcept
      -> tag_invoke_result_t<get_completion_scheduler_t, SenderT> {
    return GetCompletionScheduler(self.sender_);
  }
};

struct then_t {
  template <typename Sender, typename Func,
            std::enable_if_t<_is_sender<Sender> &&
                                 _tag_invocable_with_completion_scheduler<then_t, Sender, Func>,
                             int> = 0>
  auto operator()(Sender&& sender, Func func) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(then_t{}, std::move(scheduler), (Sender &&) sender, std::move(func));
  }

  template <typename Sender, typename Func,
            std::enable_if_t<_is_sender<Sender> &&
                                 !_tag_invocable_with_completion_scheduler<then_t, Sender, Func> &&
                                 tag_invocable<then_t, Sender, Func>,
                             int> = 0>
  auto operator()(Sender&& sender, Func func) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(then_t{}, std::move(scheduler), (Sender &&) sender, std::move(func));
  }

  template <typename Sender, typename Func,
            std::enable_if_t<_is_sender<Sender> &&
                                 !_tag_invocable_with_completion_scheduler<then_t, Sender, Func> &&
                                 !tag_invocable<then_t, Sender, Func>,
                             int> = 0>
  sender_t<Sender, Func> operator()(Sender&& sender, Func func) const {
    return {(Sender &&) sender, std::move(func)};
  }
  template <typename Func>
  _BinderBack<then_t, Func> operator()(Func func) const {
    return {{}, {}, {std::move(func)}};
  }
};

}  // namespace __then

using __then::then_t;
inline constexpr then_t Then;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_THEN_H_
