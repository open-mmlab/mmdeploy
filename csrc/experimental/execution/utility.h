// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_UTILITY_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_UTILITY_H_

#include <type_traits>
#include <utility>

#include "core/mpl/detected.h"

namespace mmdeploy {

template <typename T, typename E, typename U = void>
using _decays_to = std::enable_if_t<std::is_same<std::decay_t<T>, E>::value, U>;

template <typename... Ts>
using __decayed_tuple = std::tuple<std::decay_t<Ts>...>;

template <typename Fun, typename... As>
using __call_result_t = decltype(std::declval<Fun>()(std::declval<As>()...));

template <typename F>
struct __conv {
  F f_;
  using type = __call_result_t<F>;
  operator type() && { return ((F &&) f_)(); }
};

template <typename F>
__conv(F) -> __conv<F>;

template <typename T, typename = std::enable_if_t<std::is_destructible_v<T>>>
struct __conv_proxy {
  T v_;
  template <typename F>
  explicit __conv_proxy(F&& f) : v_(((F &&) f)()) {}
  T& operator*() noexcept { return v_; }
};

template <typename _Member, typename _Self>
_Member _Self::*__memptr(const _Self&);

template <typename _Self, typename _Member>
using __member_t = decltype((std::declval<_Self>().*__memptr<_Member>(std::declval<_Self>())));

template <typename From, typename To>
using _copy_cvref_t = __member_t<From, To>;

template <typename S, typename R>
using connect_result_t = decltype(Connect(std::declval<S>(), std::declval<R>()));

template <typename...>
struct _types
#if defined(__GNUC__) && !defined(__clang__)
{
}
#endif
;

template <typename Sender>
using _get_completion_scheduler_t = decltype(GetCompletionScheduler(std::declval<Sender>()));

template <typename Sender>
inline constexpr auto _has_completion_scheduler =
    detail::is_detected_v<_get_completion_scheduler_t, Sender>;

template <typename Sender, typename SFINAE = void>
struct _completion_signature_for {};

template <typename Sender>
struct _completion_signature_for<Sender, std::void_t<typename Sender::value_type>> {
  using type = typename Sender::value_type;
};

template <typename Sender>
using completion_signature_for_t = typename _completion_signature_for<Sender>::type;

template <typename Sender>
inline constexpr bool _is_sender = detail::is_detected_v<completion_signature_for_t, Sender>;

template <typename Sender>
inline constexpr bool _decays_to_sender = _is_sender<std::decay_t<Sender>>;

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

  template <typename Sender, std::enable_if_t<_decays_to_sender<Sender>, int> = 0>
  std::invoke_result_t<T1, std::invoke_result_t<T0, Sender>> operator()(Sender&& sender) && {
    return ((T1 &&) t1_)(((T0 &&) t0_)((Sender &&) sender));
  }

  template <typename Sender, std::enable_if_t<_decays_to_sender<Sender>, int> = 0>
  std::invoke_result_t<T1, std::invoke_result_t<T0, Sender>> operator()(Sender&& sender) const& {
    return t1_(t0_((Sender &&) sender));
  }
};

template <typename D>
struct SenderAdaptorClosure {};

template <typename T0, typename T1,
          typename = std::enable_if_t<
              std::is_base_of_v<SenderAdaptorClosure<std::decay_t<T0>>, std::decay_t<T0>> &&
              std::is_base_of_v<SenderAdaptorClosure<std::decay_t<T1>>, std::decay_t<T1>>>>
_Compose<std::decay_t<T0>, std::decay_t<T1>> operator|(T0&& t0, T1&& t1) {
  return {(T0 &&) t0, (T1 &&) t1};
}

template <typename Sender, typename Closure,
          typename = std::enable_if_t<_decays_to_sender<Sender> &&
                                      std::is_base_of_v<SenderAdaptorClosure<std::decay_t<Closure>>,
                                                        std::decay_t<Closure>>>>
std::invoke_result_t<Closure, Sender> operator|(Sender&& sender, Closure&& closure) {
  return ((Closure &&) closure)((Sender &&) sender);
}

template <typename Func, typename... As>
struct _BinderBack : SenderAdaptorClosure<_BinderBack<Func, As...>> {
  Func func_;
  std::tuple<As...> as_;

  template <typename Sender, std::enable_if_t<_decays_to_sender<Sender>, int> = 0>
  std::invoke_result_t<Func, Sender, As...> operator()(Sender&& sender) && {
    return std::apply(
        [&sender, this](As&... as) { return ((Func &&) func_)((Sender &&) sender, (As &&) as...); },
        as_);
  }

  template <typename Sender, std::enable_if_t<_decays_to_sender<Sender>, int> = 0>
  std::invoke_result_t<Func, Sender, As...> operator()(Sender&& sender) const& {
    return std::apply([&sender, this](const As&... as) { return func_((Sender &&) sender, as...); },
                      as_);
  }
};

}  // namespace __closure

using __closure::_BinderBack;

namespace __schedule {

struct schedule_t {
  template <typename Scheduler>
  auto operator()(Scheduler&& scheduler) const
      -> decltype(mmdeploySchedule((Scheduler &&) scheduler)) {
    return mmdeploySchedule((Scheduler &&) scheduler);
  }
};

}  // namespace __schedule

using __schedule::schedule_t;
inline constexpr schedule_t Schedule{};

template <typename Scheduler>
using schedule_result_t = decltype(Schedule(std::declval<Scheduler>()));

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_UTILITY_H_
