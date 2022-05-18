// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SUBMIT_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SUBMIT_H_

#include "utility.h"

namespace mmdeploy {

namespace __submit {

namespace __impl {

template <typename Sender, typename Receiver>
struct _Operation {
  struct type;
};
template <typename Sender, typename Receiver>
using operation_t = typename _Operation<Sender, remove_cvref_t<Receiver>>::type;

template <typename Sender, typename Receiver>
struct _Operation<Sender, Receiver>::type {
  struct _Receiver {
    type* op_state_;
    template <typename... As>
    friend void tag_invoke(set_value_t, _Receiver&& self, As&&... as) noexcept {
      std::unique_ptr<type> _{self.op_state_};
      return SetValue(std::move(self.op_state_->receiver_), (As &&) as...);
    }
  };
  Receiver receiver_;
  connect_result_t<Sender, _Receiver> op_state_;

  template <typename Receiver2, _decays_to<Receiver2, Receiver, int> = 0>
  type(Sender&& sender, Receiver2&& receiver)
      : receiver_((Receiver2 &&) receiver),
        op_state_(Connect((Sender &&) sender, _Receiver{this})) {}
};

}  // namespace __impl

struct __submit_t {
  template <typename Receiver, typename Sender>
  void operator()(Sender&& sender, Receiver&& receiver) const noexcept(false) {
    Start((new __impl::operation_t<Sender, Receiver>((Sender &&) sender, (Receiver &&) receiver))
              ->op_state_);
  }
};

}  // namespace __submit

using __submit::__submit_t;
inline constexpr __submit_t __Submit{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_SUBMIT_H_
