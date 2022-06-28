// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXECUTION_EXPAND_H_
#define MMDEPLOY_CSRC_EXECUTION_EXPAND_H_

#include "closure.h"
#include "concepts.h"
#include "utility.h"

namespace mmdeploy {

namespace _expand {

template <typename Sender, typename Receiver>
struct _Receiver {
  struct type {
    Receiver receiver_;
    template <class Tuple>
    friend void tag_invoke(set_value_t, type&& self, Tuple&& tup) noexcept {
      std::apply(
          [&](auto&&... args) {
            SetValue((Receiver &&) self.receiver_, (decltype(args)&&)args...);
          },
          (Tuple &&) tup);
    }
  };
};
template <typename Sender, typename Receiver>
using receiver_t = typename _Receiver<Sender, remove_cvref_t<Receiver>>::type;

template <typename Sender>
struct _Sender {
  struct type {
    using value_types = std::tuple_element_t<0, completion_signatures_of_t<Sender>>;
    Sender sender_;

    template <typename Self, typename Receiver, _decays_to<Self, type, bool> = true>
    friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver) {
      return Connect(((Self &&) self).sender_,
                     receiver_t<Sender, Receiver>{(Receiver &&) receiver});
    }
  };
};
template <typename Sender>
using sender_t = typename _Sender<remove_cvref_t<Sender>>::type;

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
