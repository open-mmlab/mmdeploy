// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_LET_VALUE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_LET_VALUE_H_

#include <optional>

#include "utility.h"

namespace mmdeploy {

namespace __let_value {

template <typename T>
using __decay_ref = std::decay_t<T>&;

template <typename Func, typename... As>
using __result_sender_t = __call_result_t<Func, __decay_ref<As>...>;

template <typename Func, typename Tuple>
struct __value_type {};

template <typename Func, typename... As>
struct __value_type<Func, std::tuple<As...>> {
  using type = __result_sender_t<Func, As...>;
};

template <typename Func, typename Tuple>
using __value_type_t = typename __value_type<Func, Tuple>::type;

template <typename CvrefSender, typename Receiver, typename Fun>
struct _Storage {
  using Sender = remove_cvref_t<CvrefSender>;
  using operation_t =
      connect_result_t<__value_type_t<Fun, completion_signatures_of_t<Sender>>, Receiver>;
  std::optional<completion_signatures_of_t<Sender>> args_;
  // workaround for MSVC v142 toolset, copy elision does not work here
  std::optional<__conv_proxy<operation_t>> proxy_;
};

template <typename CvrefSender, typename Receiver, typename Func>
struct _Operation {
  struct type;
};
template <typename CvrefSender, typename Receiver, typename Func>
using operation_t = typename _Operation<CvrefSender, remove_cvref_t<Receiver>, Func>::type;

template <typename CvrefSender, typename Receiver, typename Func>
struct _Receiver {
  struct type;
};
template <typename CvrefSender, typename Receiver, typename Func>
using receiver_t = typename _Receiver<CvrefSender, Receiver, Func>::type;

template <typename CvrefSender, typename Receiver, typename Func>
struct _Receiver<CvrefSender, Receiver, Func>::type {
  operation_t<CvrefSender, Receiver, Func>* op_state_;

  template <typename... As>
  friend void tag_invoke(set_value_t, type&& self, As&&... as) noexcept {
    auto* op_state = self.op_state_;
    auto& args = op_state->storage_.args_.emplace((As &&) as...);
    op_state->storage_.proxy_.emplace([&] {
      return Connect(std::apply(std::move(op_state->func_), args), std::move(op_state->receiver_));
    });
    Start(**op_state->storage_.proxy_);
  }
};

template <typename CvrefSender, typename Receiver, typename Func>
struct _Operation<CvrefSender, Receiver, Func>::type {
  using _receiver_t = receiver_t<CvrefSender, Receiver, Func>;

  friend void tag_invoke(start_t, type& self) noexcept { Start(self.op_state2_); }

  template <typename Receiver2>
  type(CvrefSender&& sender, Receiver2&& receiver, Func func)
      : op_state2_(Connect((CvrefSender &&) sender, _receiver_t{this})),
        receiver_((Receiver2 &&) receiver),
        func_(std::move(func)) {}

  connect_result_t<CvrefSender, _receiver_t> op_state2_;
  Receiver receiver_;
  Func func_;
  _Storage<CvrefSender, Receiver, Func> storage_;
};

template <typename Sender, typename Func>
struct _Sender {
  struct type;
};
template <typename Sender, typename Func>
using sender_t = typename _Sender<remove_cvref_t<Sender>, Func>::type;

template <typename Sender, typename Func>
struct _Sender<Sender, Func>::type {
  template <typename Self, typename Receiver>
  using _operation_t = operation_t<_copy_cvref_t<Self, Sender>, Receiver, Func>;

  using value_types =
      completion_signatures_of_t<__value_type_t<Func, completion_signatures_of_t<Sender>>>;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver)
      -> _operation_t<Self, Receiver> {
    return _operation_t<Self, Receiver>{((Self &&) self).sender_, (Receiver &&) receiver,
                                        ((Self &&) self).func_};
  }
  Sender sender_;
  Func func_;
};

using std::enable_if_t;

struct let_value_t {
  template <typename Sender, typename Func,
            enable_if_t<_is_sender<Sender> &&
                            _tag_invocable_with_completion_scheduler<let_value_t, Sender, Func>,
                        int> = 0>
  auto operator()(Sender&& sender, Func func) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(let_value_t{}, std::move(scheduler), (Sender &&) sender, std::move(func));
  }

  template <typename Sender, typename Func,
            enable_if_t<_is_sender<Sender> &&
                            _tag_invocable_with_completion_scheduler<let_value_t, Sender, Func> &&
                            tag_invocable<let_value_t, Sender, Func>,
                        int> = 0>
  auto operator()(Sender&& sender, Func func) const {
    return tag_invoke(let_value_t{}, (Sender &&) sender, std::move(func));
  }

  template <typename Sender, typename Func,
            enable_if_t<_is_sender<Sender> &&
                            !_tag_invocable_with_completion_scheduler<let_value_t, Sender, Func> &&
                            !tag_invocable<let_value_t, Sender>,
                        int> = 0>
  sender_t<Sender, Func> operator()(Sender&& sender, Func func) const {
    return {(Sender &&) sender, std::move(func)};
  }
  template <typename Func>
  _BinderBack<let_value_t, Func> operator()(Func func) const {
    return {{}, {}, {std::move(func)}};
  }
};

}  // namespace __let_value

using __let_value::let_value_t;
inline constexpr let_value_t LetValue{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_LET_VALUE_H_
