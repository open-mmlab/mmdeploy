// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/__utility.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_UTILITY_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_UTILITY_H_

#include <type_traits>
#include <utility>

#include "mmdeploy/core/mpl/detected.h"
#include "tag_invoke.h"

#define MMDEPLOY_REQUIRES(...) std::enable_if_t<__VA_ARGS__, int> = 0

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
__conv(F)->__conv<F>;

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

namespace __schedule {

struct schedule_t {
  template <typename Scheduler, std::enable_if_t<tag_invocable<schedule_t, Scheduler>, int> = 0>
  auto operator()(Scheduler&& scheduler) const -> tag_invoke_result_t<schedule_t, Scheduler> {
    return tag_invoke(schedule_t{}, (Scheduler &&) scheduler);
  }
};

}  // namespace __schedule

using __schedule::schedule_t;
inline constexpr schedule_t Schedule{};

template <typename Scheduler>
using schedule_result_t = decltype(std::declval<schedule_t>()(std::declval<Scheduler>()));

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_UTILITY_H_
