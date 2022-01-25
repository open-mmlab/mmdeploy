//// Copyright (c) OpenMMLab. All rights reserved.
//
//#ifndef MMDEPLOY_SRC_EXPERIMENTAL_COLLECTION_H_
//#define MMDEPLOY_SRC_EXPERIMENTAL_COLLECTION_H_
//
//#include "token.h"
//
// namespace mmdeploy {
//
// class Collection {
// public:
//  template <typename... Args>
//  friend Collection& operator<<(Collection& c, const Token<Args...>& value) {
//    c.put(value);
//    return c;
//  }
//
//  template <typename... Args>
//  friend const Collection& operator>>(const Collection& c, Token<Args...>& value) {
//    c.get(value);
//    return c;
//  }
//
//  template <typename T>
//  Result<T> maybe() const {
//    T token;
//    if (get(token)) {
//      return token;
//    }
//    return Status(eFail);
//  }
//
// private:
//  std::vector<std::string> keys_;
//  std::vector<std::shared_ptr<void>> values_;
//
//  template <typename... Args>
//  void put(const Token<Args...>& value) {
//    keys_.push_back(Token<Args...>::key());
//    values_.push_back(std::make_shared<Token<Args...>>(value));
//  }
//
//  template <typename... Args>
//  bool get(Token<Args...>& value) const {
//    for (int i = 0; i < keys_.size(); ++i) {
//      if (keys_[i] == Token<Args...>::key()) {
//        value = *static_cast<Token<Args...>*>(values_[i].get());
//        return true;
//      }
//    }
//    return false;
//  }
//};
//
// namespace detail {
//
// template <typename T>
// struct function_traits {
//  template <typename R, typename... As>
//  static std::tuple<As...> get_args(std::function<R(As...)>);
//
//  template <typename R, typename... Args>
//  static R get_ret(std::function<R(Args...)>);
//
//  using args_t = decltype(get_args(std::function{std::declval<T>()}));
//  using ret_t = decltype(get_ret(std::function{std::declval<T>()}));
//};
//
//// TODO: obtain first error
//// TODO: combine all errors
// template <typename F, typename... Args, typename Ret = std::invoke_result_t<F, Args...>>
// Result<Ret> Apply(F&& f, const Result<Args>&... args) {
//   if ((... && args)) {
//     return std::invoke(std::forward<F>(f), args.value()...);
//   }
//   return Status(eFail);
// }
//
// template <typename F, typename... Args, typename Ret = std::invoke_result_t<F, Args...>>
// Result<Ret> ApplyImpl(F&& f, const Collection& c, std::tuple<Args...>*) {
//   return Apply(std::forward<F>(f), c.maybe<Args>()...);
// }
//
// }  // namespace detail
//
// template <typename F, typename Args = typename detail::function_traits<F>::args_t>
// decltype(auto) Apply(F&& f, const Collection& c) {
//   return detail::ApplyImpl(std::forward<F>(f), c, std::add_pointer_t<Args>{});
// }
//
// }  // namespace mmdeploy
//
//#endif  // MMDEPLOY_SRC_EXPERIMENTAL_COLLECTION_H_
