// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_MODULE_ADAPTER_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_MODULE_ADAPTER_H_

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/mpl/type_traits.h"

namespace mmdeploy {

namespace module_detail {

template <typename T>
struct is_tuple : std::false_type {};

template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename T>
inline constexpr auto is_tuple_v = is_tuple<T>::value;

template <typename Ret, typename... Args>
struct InvokeImpl {
  template <typename F, typename... Ts>
  static Result<Value> apply(F&& f, const Value& params, Ts&&... ts) {
    std::tuple<uncvref_t<Args>...> args;
    try {
      from_value(params, args);
      auto ret = apply_impl(std::forward<F>(f), std::move(args), std::index_sequence_for<Args...>{},
                            std::forward<Ts>(ts)...);
      return make_ret_val(std::move(ret));
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("unhandled exception: {}", e.what());
      return Status(eFail);
    } catch (...) {
      return Status(eFail);
    }
  }

  template <typename F, typename Tuple, size_t... Is, typename... Ts>
  static decltype(auto) apply_impl(F&& f, Tuple&& tuple, std::index_sequence<Is...>, Ts&&... ts) {
    return std::invoke(std::forward<F>(f), std::forward<Ts>(ts)...,
                       std::get<Is>(std::forward<Tuple>(tuple))...);
  }

  template <typename T, typename T0 = uncvref_t<T>>
  static Result<Value> make_ret_val(T&& ret) {
    if constexpr (module_detail::is_tuple_v<T0>) {
      return to_value(std::forward<T>(ret));
    } else if constexpr (is_result_v<T0>) {
      return ret ? make_ret_val(std::forward<T>(ret).value()) : std::forward<T>(ret).as_failure();
    } else {
      return make_ret_val(std::forward_as_tuple(std::forward<T>(ret)));
    }
  }
};

// function pointer
template <typename Ret, typename... Args>
Result<Value> Invoke(Ret (*f)(Args...), const Value& args) {
  return InvokeImpl<Ret, Args...>::apply(f, args);
}

// member function pointer
template <typename Ret, typename C, typename... Args>
Result<Value> Invoke(Ret (C::*f)(Args...) const, C* inst, const Value& args) {
  return InvokeImpl<Ret, Args...>::apply(f, args, inst);
}
template <typename Ret, typename C, typename... Args>
Result<Value> Invoke(Ret (C::*f)(Args...), C* inst, const Value& args) {
  return InvokeImpl<Ret, Args...>::apply(f, args, inst);
}

// function object
template <typename T, typename C = std::remove_reference_t<T>,
          typename = std::void_t<decltype(&C::operator())>>
Result<Value> Invoke(T&& t, const Value& args) {
  return Invoke(&C::operator(), &t, args);
}

template <typename T>
struct IsPointer : std::false_type {};
template <typename R, typename... Args>
struct IsPointer<R (*)(Args...)> : std::false_type {};
template <typename T>
struct IsPointer<std::shared_ptr<T>> : std::true_type {};
template <typename T, typename D>
struct IsPointer<std::unique_ptr<T, D>> : std::true_type {};
template <typename T>
struct IsPointer<T*> : std::true_type {};

template <typename T, typename SFINAE = void>
struct AccessPolicy {
  static constexpr auto apply = [](auto& x) -> decltype(auto) { return x; };
};
template <typename T>
struct AccessPolicy<T, std::enable_if_t<IsPointer<T>::value>> {
  static constexpr auto apply = [](auto& x) -> decltype(auto) { return *x; };
};

template <typename T, typename A = AccessPolicy<T>>
class Task : public Module {
 public:
  explicit Task(T task) : task_(std::move(task)) {}

  Result<Value> Process(const Value& arg) override {
    return module_detail::Invoke(A::apply(task_), arg);
  }

 private:
  T task_;
};

template <typename T>
std::unique_ptr<Module> CreateTask(T&& x) {
  return std::unique_ptr<Module>(new Task{std::forward<T>(x)});
}

template <typename T>
auto MakeTask(T&& x) {
  return Task(std::forward<T>(x));
}

}  // namespace module_detail

using module_detail::CreateTask;

using module_detail::MakeTask;

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_MODULE_ADAPTER_H_
