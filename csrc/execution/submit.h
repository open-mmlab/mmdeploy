// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SUBMIT_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SUBMIT_H_

#include "utility.h"

namespace mmdeploy {

namespace __submit {

namespace __impl {

template <typename Predecessor, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Predecessor, typename Receiver>
using Operation = typename _Operation<Predecessor, remove_cvref_t<Receiver>>::type;

template <typename Predecessor, typename Receiver>
struct _Operation<Predecessor, Receiver>::type {
  struct _Receiver {
    type* op_state_;
    template <typename... As>
    friend void SetValue(_Receiver&& self, As&&... as) noexcept {
      std::unique_ptr<type> _{self.op_state_};
      return SetValue(std::move(self.op_state_->receiver_), (As &&) as...);
    }
  };
  Receiver receiver_;
  connect_result_t<Predecessor, _Receiver> op_state_;

  template <typename Receiver2, _decays_to<Receiver2, Receiver, int> = 0>
  type(Predecessor&& pred, Receiver2&& receiver)
      : receiver_((Receiver2 &&) receiver),
        op_state_(Connect((Predecessor &&) pred, _Receiver{this})) {}
};

}  // namespace __impl

struct __submit_t {
  template <typename Receiver, typename Sender>
  void operator()(Sender&& sender, Receiver&& receiver) const noexcept(false) {
    Start((new __impl::Operation<Sender, Receiver>((Sender &&) sender, (Receiver &&) receiver))
              ->op_state_);
  }
};

}  // namespace __submit

using __submit::__submit_t;
inline constexpr __submit_t __Submit{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SUBMIT_H_
