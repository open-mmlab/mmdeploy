// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_EXPAND_H_
#define MMDEPLOY_CSRC_EXECUTION_EXPAND_H_

#include "closure.h"
#include "concepts.h"
#include "utility.h"

namespace mmdeploy {

namespace _expand {

template <typename Sender, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Sender, typename Receiver>
using operation_t = typename _Operation<Sender, remove_cvref_t<Receiver>>::type;

template <typename Sender, typename Receiver>
struct _Receiver {
  struct type;
};
template <typename Sender, typename Receiver>
using receiver_t = typename _Receiver<Sender, Receiver>::type;

template <typename Sender, typename Receiver>
struct _Receiver<Sender, Receiver>::type {
  operation_t<Sender, Receiver>* op_state_;

  template <class Tuple>
  friend void tag_invoke(set_value_t, type&& self, Tuple&& tup) noexcept {
    std::apply(
        [&](auto&&... args) {
          SetValue((Receiver &&) self.op_state_->receiver_, (decltype(args)&&)args...);
        },
        (Tuple &&) tup);
  }
};

template <typename Sender, typename Receiver>
struct _Operation<Sender, Receiver>::type {
  connect_result_t<Sender, receiver_t<Sender, Receiver>> op_state2_;
  Receiver receiver_;

  template <typename Sender2>
  type(Sender2&& sender, Receiver&& receiver)
      : op_state2_(Connect((Sender2 &&) sender, receiver_t<Sender, Receiver>{this})),
        receiver_((Receiver &&) receiver) {}

  friend void tag_invoke(start_t, type& op_state) { Start(op_state.op_state2_); }
};

template <typename Sender>
struct _Sender {
  struct type;
};
template <typename Sender>
using sender_t = typename _Sender<remove_cvref_t<Sender>>::type;

template <typename Sender>
struct _Sender<Sender>::type {
  using value_types = std::tuple_element_t<0, completion_signatures_of_t<Sender>>;
  Sender sender_;

  template <typename Self, typename Receiver, _decays_to<Self, type, bool> = true>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver)
      -> operation_t<Sender, Receiver> {
    return operation_t<Sender, Receiver>(((Self &&) self).sender_, (Receiver &&) receiver);
  }
};

struct expand_t {
  template <typename Sender, std::enable_if_t<_is_sender<Sender>, int> = 0>
  auto operator()(Sender&& sender) const {
    return sender_t<Sender>{(Sender &&) sender};
  }
  _BinderBack<expand_t> operator()() const { return {{}, {}, {}}; }
};

}  // namespace _expand

using _expand::expand_t;
inline constexpr expand_t Expand{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXECUTION_EXPAND_H_
