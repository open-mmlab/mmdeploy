// Copyright (c) OpenMMLab. All rights reserved.

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

  template <class... Args>
  friend void SetValue(type&& self, Args&&... args) {
    SetValue(std::move(self.receiver_), std::invoke(std::move(self.func_), (Args &&) args...));
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
  using _ret_type = decltype(std::apply(std::declval<Func>(),
                                        std::declval<completion_signatures_of_t<Sender>>()));

  using value_types =
      std::conditional_t<std::is_void_v<_ret_type>, std::tuple<>, std::tuple<_ret_type>>;

  Sender sender_;
  Func func_;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver) {
    return Connect(((Self &&) self).sender_,
                   receiver_t<Receiver, Func>{(Receiver &&) receiver, std::move(self.func_)});
  }

  friend auto tag_invoke(get_completion_scheduler_t, const _Sender& self) noexcept {
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
