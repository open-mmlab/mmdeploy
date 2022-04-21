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
  using Sender = std::decay_t<CvrefSender>;
  using operation_t =
      connect_result_t<__value_type_t<Fun, completion_signature_for_t<Sender>>, Receiver>;
  std::optional<completion_signature_for_t<Sender>> args_;
  // workaround for MSVC v142 toolset, copy elision does not work here
  std::optional<__conv_proxy<operation_t>> proxy_;
};

template <typename CvrefSender, typename Receiver, typename Func>
struct _Operation {
  struct type;
};
template <typename CvrefSender, typename Receiver, typename Func>
using Operation =
    typename _Operation<CvrefSender, std::decay_t<Receiver>, std::decay_t<Func>>::type;

template <typename CvrefSender, typename Receiver, typename Func>
struct _Receiver {
  struct type;
};
template <typename CvrefSender, typename Receiver, typename Func>
using receiver_t =
    typename _Receiver<CvrefSender, std::decay_t<Receiver>, std::decay_t<Func>>::type;

template <typename CvrefSender, typename Receiver, typename Func>
struct _Receiver<CvrefSender, Receiver, Func>::type {
  Operation<CvrefSender, Receiver, Func>* op_state_;

  template <typename... As>
  friend void SetValue(type&& self, As&&... as) noexcept {
    //    using operation_t = typename _Storage<CvrefSender, Receiver, Func>::operation_t;
    auto* op_state = self.op_state_;
    auto& args = op_state->storage_.args_.emplace((As &&) as...);
    op_state->storage_.proxy_.emplace([&] {
      return Connect(std::apply(std::move(op_state->func_), args), std::move(op_state->receiver_));
    });
    Start(**op_state->storage_.proxy_);
  }
};

template <typename CvrefPredecessor, typename Receiver, typename Func>
struct _Operation<CvrefPredecessor, Receiver, Func>::type {
  using _receiver_t = receiver_t<CvrefPredecessor, Receiver, Func>;

  friend void Start(type& self) noexcept { Start(self.op_state2_); }

  template <typename Receiver2>
  type(CvrefPredecessor&& pred, Receiver2&& receiver, Func func)
      : op_state2_(Connect((CvrefPredecessor &&) pred, _receiver_t{this})),
        receiver_((Receiver2 &&) receiver),
        func_(std::move(func)) {}

  connect_result_t<CvrefPredecessor, _receiver_t> op_state2_;
  Receiver receiver_;
  Func func_;
  _Storage<CvrefPredecessor, Receiver, Func> storage_;
};

template <typename Predecessor, typename Func>
struct _Sender {
  struct type;
};
template <typename Predecessor, typename Func>
using Sender = typename _Sender<std::decay_t<Predecessor>, std::decay_t<Func>>::type;

template <typename Predecessor, typename Func>
struct _Sender<Predecessor, Func>::type {
  template <typename Self, typename Receiver>
  using operation_t = Operation<_copy_cvref_t<Self, Predecessor>, Receiver, Func>;

  using value_type =
      completion_signature_for_t<__value_type_t<Func, completion_signature_for_t<Predecessor>>>;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver) -> operation_t<Self, Receiver> {
    return operation_t<Self, Receiver>{((Self &&) self).pred_, (Receiver &&) receiver,
                                       ((Self &&) self).func_};
  }
  Predecessor pred_;
  Func func_;
};

struct let_value_t {
  template <typename Predecessor, typename Func,
            std::enable_if_t<_decays_to_sender<Predecessor>, int> = 0>
  Sender<Predecessor, Func> operator()(Predecessor&& pred, Func&& func) const {
    return {(Predecessor &&) pred, (Func &&) func};
  }
  template <typename Func>
  _BinderBack<let_value_t, std::decay_t<Func>> operator()(Func&& func) const {
    return {{}, {}, {(Func &&) func}};
  }
};

}  // namespace __let_value

using __let_value::let_value_t;
inline constexpr let_value_t LetValue{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_LET_VALUE_H_
