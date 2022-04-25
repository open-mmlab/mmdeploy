// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_CONCEPTS_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_CONCEPTS_H_

#include "tag_invoke.h"

namespace mmdeploy {

namespace _get_completion_signatures {

struct get_completion_signatures_t {
  template <typename Sender, typename ValueTypes = typename remove_cvref_t<Sender>::value_types>
  constexpr identity<ValueTypes> operator()(Sender&& sender) const noexcept {
    return {};
  }
};

}  // namespace _get_completion_signatures

using _get_completion_signatures::get_completion_signatures_t;
inline constexpr get_completion_signatures_t GetCompletionSignatures{};

template <typename Sender>
inline constexpr bool _is_sender = std::is_invocable_v<get_completion_signatures_t, Sender>&&
    std::is_move_constructible_v<remove_cvref_t<Sender>>;

// GetCompletionSignatures is expected to return identity<std::tuple<Types...>>;
template <typename Sender>
using completion_signatures_of_t =
    typename std::invoke_result_t<get_completion_signatures_t, Sender>::type;

namespace _set_value {
struct set_value_t {
  template <typename Receiver, typename... Args,
            std::enable_if_t<is_tag_invocable_v<set_value_t, Receiver, Args...>, int> = 0>
  void operator()(Receiver&& receiver, Args&&... args) const noexcept {
    static_assert(is_nothrow_tag_invocable_v<set_value_t, Receiver, Args...>);
    (void)tag_invoke(set_value_t{}, (Receiver &&) receiver, (Args &&) args...);
  }
  template <typename Receiver, typename... Args,
            std::enable_if_t<!is_tag_invocable_v<set_value_t, Receiver, Args...>, int> = 0>
  auto operator()(Receiver&& receiver, Args&&... args) const noexcept
      -> decltype(((Receiver &&) receiver).SetValue((Args &&) args...)) {
    static_assert(noexcept(((Receiver &&) receiver).SetValue((Args &&) args...)),
                  "SetValue must be noexcept");
    return ((Receiver &&) receiver).SetValue((Args &&) args...);
  }
};

}  // namespace _set_value

// using _set_value::set_value_t;
// inline constexpr set_value_t SetValue{};

// template <typename Receiver>
// inline constexpr bool is_receiver_v = std::is_move_constructible_v<remove_cvref<Receiver> >&&
//     std::is_constructible_v<remove_cvref<Receiver>, Receiver>;
//
// template <typename Receiver, typename... Args>
// inline constexpr bool is_receiver_of_v =
//     is_tag_invocable_v<set_value_t, remove_cvref<Receiver>&&, Args...>;

namespace _start {

struct start_t {
  template <typename Operation, std::enable_if_t<is_tag_invocable_v<start_t, Operation>, int> = 0>
  void operator()(Operation& op_state) const noexcept {
    static_assert(is_nothrow_tag_invocable_v<start_t, Operation&>);
    (void)tag_invoke(start_t{}, op_state);
  }
  template <typename Operation, std::enable_if_t<!is_tag_invocable_v<start_t, Operation>, int> = 0>
  auto operator()(Operation& op) const noexcept -> decltype(op.Start()) {
    static_assert(noexcept(op.Start()), "Start() must be noexcept");
    return op.Start();
  }
};

}  // namespace _start

// using _start::start_t;
// inline constexpr start_t Start{};

// template <typename Operation>
// inline constexpr bool is_operation_state_v =                           //
//     std::is_destructible_v<Operation>&& std::is_object_v<Operation>&&  //
//         std::is_invocable_v<start_t, Operation&>;                      //

namespace _connect {

struct connect_t {
  template <typename Sender, typename Receiver,
            std::enable_if_t<is_tag_invocable_v<connect_t, Sender, Receiver>, int> = 0>
  auto operator()(Sender&& sender, Receiver&& receiver) const
      -> tag_invoke_result_t<connect_t, Sender, Receiver> {
    return tag_invoke(connect_t{}, (Sender &&) sender, (Receiver &&) receiver);
  }
  template <typename Sender, typename Receiver,
            std::enable_if_t<!is_tag_invocable_v<connect_t, Sender, Receiver>, int> = 0>
  auto operator()(Sender&& sender, Receiver&& receiver) const
      -> decltype(((Sender &&) sender).Connect((Receiver &&) receiver)) {
    return ((Sender &&) sender).Connect((Receiver &&) receiver);
  }
};

}  // namespace _connect

// using _connect::connect_t;
// inline constexpr connect_t Connect{};

namespace _get_completion_scheduler {

struct get_completion_scheduler_t {
  template <
      typename Sender,
      std::enable_if_t<is_tag_invocable_v<get_completion_scheduler_t, const Sender&>, int> = 0>
  auto operator()(const Sender& sender) const noexcept
      -> tag_invoke_result_t<get_completion_scheduler_t, const Sender&> {
    return tag_invoke(get_completion_scheduler_t{}, sender);
  }
};

}  // namespace _get_completion_scheduler

using _get_completion_scheduler::get_completion_scheduler_t;
inline constexpr get_completion_scheduler_t GetCompletionScheduler{};

template <typename Sender>
inline constexpr bool _has_completion_scheduler_v =
    std::is_invocable_v<get_completion_scheduler_t, Sender>;

template <typename Sender>
struct _has_completion_scheduler : std::bool_constant<_has_completion_scheduler_v<Sender>> {};

template <typename Sender>
using _completion_scheduler_for = std::invoke_result_t<get_completion_scheduler_t, Sender>;

// template <typename Func, typename Sender, typename... Args>
// inline constexpr bool _tag_invocable_with_completion_scheduler =
//     _has_completion_scheduler_v<Sender>&&
//         std::is_invocable_v<Func, _completion_scheduler_for<Sender>, Sender, Args...>;

// template <typename Func, typename Sender, typename... Args>
// inline constexpr bool _tag_invocable_with_completion_scheduler = std::conjunction_v<
//     _has_completion_scheduler<Sender>,
//     _defer_args<std::is_invocable, identity<Func>, _defer<_completion_scheduler_for, Sender>,
//                 identity<Sender>, identity<Args>...>>;

namespace impl {

template <typename Func, typename Sender, typename TArgs, typename SFINAE = void>
struct _tag_invocable_with_completion_scheduler : std::false_type {};

template <typename Func, typename Sender, typename... Args>
struct _tag_invocable_with_completion_scheduler<
    Func, Sender, std::tuple<Args...>, std::enable_if_t<_has_completion_scheduler_v<Sender>>>
    : is_tag_invocable<Func, _completion_scheduler_for<Sender>, Sender, Args...> {};

}  // namespace impl

template <typename Func, typename Sender, typename... Args>
inline constexpr bool _tag_invocable_with_completion_scheduler =
    impl::_tag_invocable_with_completion_scheduler<Func, Sender, std::tuple<Args...>>::value;

template <typename T, typename SFINAE = void>
struct _is_range : std::false_type {};

template <typename T>
struct _is_range<T,
                 std::void_t<decltype(std::begin(std::declval<T>()), std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool _is_range_v = _is_range<T>::value;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_CONCEPTS_H_
