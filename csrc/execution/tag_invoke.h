// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/facebookexperimental/libunifex/blob/main/include/unifex/tag_invoke.hpp
#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TAG_INVOKE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TAG_INVOKE_H_

#include "type_traits.h"

namespace mmdeploy {

namespace _tag_invoke {

void tag_invoke();

struct _fn {
  template <typename CPO, typename... Args>
  constexpr auto operator()(CPO cpo, Args&&... args) const
      noexcept(noexcept(tag_invoke((CPO &&) cpo, (Args &&) args...)))
          -> decltype(tag_invoke((CPO &&) cpo, (Args &&) args...)) {
    return tag_invoke((CPO &&) cpo, (Args &&) args...);
  }
};

template <typename CPO, typename... Args>
using tag_invoke_result_t = decltype(tag_invoke(std::declval<CPO>(), std::declval<Args>()...));

using yes_type = char;
using no_type = char (&)[2];

template <typename CPO, typename... Args>
auto try_tag_invoke(int) noexcept(noexcept(tag_invoke(std::declval<CPO>(),
                                                      std::declval<Args>()...)))
    -> decltype((void)tag_invoke(std::declval<CPO&&>(), std::declval<Args>()...), yes_type{});

template <typename CPO, typename... Args>
no_type try_tag_invoke(...) noexcept(false);

template <template <typename...> class T, typename... Args>
struct defer {
  using type = T<Args...>;
};

struct empty {};

}  // namespace _tag_invoke

namespace _tag_invoke_cpo {
inline constexpr _tag_invoke::_fn tag_invoke{};
}
using namespace _tag_invoke_cpo;

template <auto& CPO>
using tag_t = std::remove_const_t<std::remove_reference_t<decltype(CPO)>>;

using _tag_invoke::tag_invoke_result_t;

template <typename CPO, typename... Args>
inline constexpr bool is_tag_invocable_v = (sizeof(_tag_invoke::try_tag_invoke<CPO, Args...>(0))) ==
                                           (sizeof(_tag_invoke::yes_type));

template <typename CPO, typename... Args>
struct tag_invoke_result
    : std::conditional_t<is_tag_invocable_v<CPO, Args...>,
                         _tag_invoke::defer<tag_invoke_result, CPO, Args...>, _tag_invoke::empty> {
};

template <typename CPO, typename... Args>
using is_tag_invocable = std::bool_constant<is_tag_invocable_v<CPO, Args...>>;

template <typename CPO, typename... Args>
inline constexpr bool is_nothrow_tag_invocable_v =
    noexcept(_tag_invoke::try_tag_invoke<CPO, Args...>(0));

template <typename CPO, typename... Args>
using is_nothrow_tag_invocable = std::bool_constant<is_nothrow_tag_invocable_v<CPO, Args...>>;

template <typename CPO, typename... Args>
inline constexpr bool tag_invocable = (sizeof(_tag_invoke::try_tag_invoke<CPO, Args...>(0)) ==
                                       sizeof(_tag_invoke::yes_type));

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_TAG_INVOKE_H_
