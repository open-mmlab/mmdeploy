// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_THEN_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_THEN_H_

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

template <typename Predecessor, typename Func>
struct _Sender {
  struct type;
};
template <typename Predecessor, typename Func>
using Sender = typename _Sender<std::decay_t<Predecessor>, std::decay_t<Func>>::type;

template <typename Predecessor, typename Func>
struct _Sender<Predecessor, Func>::type {
  using _ret_type = decltype(std::apply(std::declval<Func>(),
                                        std::declval<completion_signature_for_t<Predecessor>>()));

  using value_type =
      std::conditional_t<std::is_void_v<_ret_type>, std::tuple<>, std::tuple<_ret_type>>;

  Predecessor pred_;
  Func func_;

  template <class Self, class Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver) {
    return Connect(((Self &&) self).pred_,
                   receiver_t<Receiver, Func>{(Receiver &&) receiver, std::move(self.func_)});
  }

  template <class Sender = Predecessor>
  friend auto GetCompletionScheduler(const _Sender& self) noexcept
      -> decltype(GetCompletionScheduler(std::declval<Sender>())) {
    return GetCompletionScheduler(self.pred_);
  }
};

struct then_t {
  template <class Predecessor, class Func>
  Sender<Predecessor, Func> operator()(Predecessor&& pred, Func func) const {
    return {(Predecessor &&) pred, std::move(func)};
  }
  template <class Fun>
  _BinderBack<then_t, Fun> operator()(Fun fun) const {
    return {{}, {}, {std::move(fun)}};
  }
};

}  // namespace __then

using __then::then_t;
inline constexpr then_t Then;

}

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_THEN_H_
