// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/facebookexperimental/libunifex/blob/main/include/unifex/type_traits.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_TRAITS_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_TRAITS_H_

#include <type_traits>

namespace mmdeploy {

/////////////////////////////////////////////////////////
// remove_cvref without handling volatile
template <typename T>
struct remove_cvref {
  using type = T;
};
template <typename T>
struct remove_cvref<const T> {
  using type = T;
};
template <typename T>
struct remove_cvref<T&> {
  using type = T;
};
template <typename T>
struct remove_cvref<const T&> {
  using type = T;
};
template <typename T>
struct remove_cvref<T&&> {
  using type = T;
};
template <typename T>
struct remove_cvref<const T&&> {
  using type = T;
};

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <typename Fn, typename... Args>
using callable_result_t = decltype(std::declval<Fn&&>()(std::declval<Args&&>()...));

namespace _is_callable {
struct yes_type {
  char dummy;
};
struct no_type {
  char dummy[2];
};
static_assert(sizeof(yes_type) != sizeof(no_type));

template <typename Fn, typename... Args, typename = callable_result_t<Fn, Args...>>
yes_type _try_call(Fn (*)(Args...)) noexcept(
    noexcept(std::declval<Fn&&>()(std::declval<Args&&>()...)));
no_type _try_call(...) noexcept(false);

}  // namespace _is_callable

template <typename Fn, typename... Args>
inline constexpr bool is_callable_v =
    sizeof(decltype(_is_callable::_try_call(static_cast<Fn (*)(Args...)>(nullptr)))) ==
    sizeof(_is_callable::yes_type);

template <typename Fn, typename... Args>
inline constexpr bool is_nothrow_callable_v =
    noexcept(_is_callable::_try_call(static_cast<Fn (*)(Args...)>(nullptr)));
}  // namespace mmdeploy

template <template <typename...> class T, typename... Args>
struct _defer {
  using type = T<Args...>;
};

template <template <typename...> class T, typename... Args>
struct _defer_args {
  using type = T<typename Args::type...>;
};

template <typename T>
struct identity {
  using type = T;
};

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TYPE_TRAITS_H_
