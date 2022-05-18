// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#include <utility>

#include "concepts.h"
#include "utility.h"

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_CLOSURE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_CLOSURE_H_

namespace mmdeploy {

namespace __closure {

template <class D>
struct SenderAdaptorClosure;

}  // namespace __closure

using __closure::SenderAdaptorClosure;

namespace __closure {

template <typename T0, typename T1>
struct _Compose : SenderAdaptorClosure<_Compose<T0, T1>> {
  T0 t0_;
  T1 t1_;

  template <typename Sender, std::enable_if_t<_is_sender<Sender>, int> = 0>
  std::invoke_result_t<T1, std::invoke_result_t<T0, Sender>> operator()(Sender&& sender) && {
    return ((T1 &&) t1_)(((T0 &&) t0_)((Sender &&) sender));
  }

  template <typename Sender, std::enable_if_t<_is_sender<Sender>, int> = 0>
  std::invoke_result_t<T1, std::invoke_result_t<T0, Sender>> operator()(Sender&& sender) const& {
    return t1_(t0_((Sender &&) sender));
  }
};

template <typename D>
struct SenderAdaptorClosure {};

template <typename T0, typename T1,
          typename = std::enable_if_t<
              std::is_base_of_v<SenderAdaptorClosure<remove_cvref_t<T0>>, remove_cvref_t<T0>> &&
              std::is_base_of_v<SenderAdaptorClosure<remove_cvref_t<T1>>, remove_cvref_t<T1>>>>
_Compose<remove_cvref_t<T0>, remove_cvref_t<T1>> operator|(T0&& t0, T1&& t1) {
  return {(T0 &&) t0, (T1 &&) t1};
}

template <typename Sender, typename Closure,
          typename = std::enable_if_t<
              _is_sender<Sender> && std::is_base_of_v<SenderAdaptorClosure<remove_cvref_t<Closure>>,
                                                      remove_cvref_t<Closure>>>>
std::invoke_result_t<Closure, Sender> operator|(Sender&& sender, Closure&& closure) {
  return ((Closure &&) closure)((Sender &&) sender);
}

template <typename Func, typename... As>
struct _BinderBack : SenderAdaptorClosure<_BinderBack<Func, As...>> {
  Func func_;
  std::tuple<As...> as_;

  template <typename Sender, std::enable_if_t<_is_sender<Sender>, int> = 0>
  std::invoke_result_t<Func, Sender, As...> operator()(Sender&& sender) && {
    return std::apply(
        [&sender, this](As&... as) { return ((Func &&) func_)((Sender &&) sender, (As &&) as...); },
        as_);
  }

  template <typename Sender, std::enable_if_t<_is_sender<Sender>, int> = 0>
  std::invoke_result_t<Func, Sender, As...> operator()(Sender&& sender) const& {
    return std::apply([&sender, this](const As&... as) { return func_((Sender &&) sender, as...); },
                      as_);
  }
};

}  // namespace __closure

using __closure::_BinderBack;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_CLOSURE_H_
