// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_MODULE_ADAPTER_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_MODULE_ADAPTER_H_

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/mpl/type_traits.h"

namespace mmdeploy {

namespace module_adapter {

template <typename T>
struct is_tuple : std::false_type {};

template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename T>
inline constexpr auto is_tuple_v = is_tuple<T>::value;

template <typename... Args>
struct InvokeImpl {
  template <typename F>
  static Result<Value> apply(F&& f, const Value& args) {
    try {
      using ArgsType = std::tuple<uncvref_t<Args>...>;
      return make_ret_val(std::apply((F &&) f, from_value<ArgsType>(args)));
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("unhandled exception: {}", e.what());
      return Status(eFail);
    } catch (...) {
      return Status(eFail);
    }
  }

  template <typename T, typename T0 = uncvref_t<T>>
  static Result<Value> make_ret_val(T&& ret) {
    if constexpr (is_tuple_v<T0>) {
      return to_value(std::forward<T>(ret));
    } else if constexpr (is_result_v<T0>) {
      return ret ? make_ret_val(std::forward<T>(ret).value()) : std::forward<T>(ret).as_failure();
    } else {
      return make_ret_val(std::forward_as_tuple(std::forward<T>(ret)));
    }
  }
};

// match function pointer
template <typename Ret, typename... Args>
Result<Value> Invoke(Ret (*f)(Args...), const Value& args) {
  return InvokeImpl<Args...>::apply(f, args);
}

// match member function pointer `&C::operator()`
template <typename Ret, typename C, typename F, typename... Args>
Result<Value> Invoke(Ret (C::*)(Args...) const, const F& f, const Value& args) {
  return InvokeImpl<Args...>::apply(f, args);
}
template <typename Ret, typename C, typename F, typename... Args>
Result<Value> Invoke(Ret (C::*)(Args...), F& f, const Value& args) {
  return InvokeImpl<Args...>::apply(f, args);
}

// match function object
template <typename F, typename C = std::remove_reference_t<F>,
          typename = std::void_t<decltype(&C::operator())>>
Result<Value> Invoke(F&& f, const Value& args) {
  return Invoke(&C::operator(), (F &&) f, args);
}

template <typename F>
Result<Value> Invoke(const std::unique_ptr<F>& f, const Value& args) {
  return Invoke(*f, args);
}
template <typename F>
Result<Value> Invoke(const std::shared_ptr<F>& f, const Value& args) {
  return Invoke(*f, args);
}

template <typename Func>
class Task : public Module {
 public:
  explicit Task(Func func) : func_(std::move(func)) {}

  Result<Value> Process(const Value& arg) override {
    return ::mmdeploy::module_adapter::Invoke(func_, arg);
  }

 private:
  Func func_;
};

template <typename T>
std::unique_ptr<Module> CreateTask(T&& x) {
  return std::unique_ptr<Module>(new Task{std::forward<T>(x)});
}

template <typename T>
auto MakeTask(T&& x) {
  return Task(std::forward<T>(x));
}

}  // namespace module_adapter

using module_adapter::CreateTask;
using module_adapter::MakeTask;

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_MODULE_ADAPTER_H_
