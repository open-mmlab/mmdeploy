//// Copyright (c) OpenMMLab. All rights reserved.
//
//#ifndef MMDEPLOY_SRC_TOKEN_TOKEN_H_
//#define MMDEPLOY_SRC_TOKEN_TOKEN_H_
//
//#include <cstdint>
//#include <memory>
//#include <string>
//#include <type_traits>
//#include <utility>
//#include <vector>
//
//#include "core/status_code.h"
//
// namespace mmdeploy {
//
// namespace token {
//
// template <char... cs>
// using String = std::integer_sequence<char, cs...>;
//
//// this is a GCC only extension
// template <typename T, T... cs>
// constexpr String<cs...> operator""_ts() {
//   return {};
// }
//
// template <char... cs>
// const char* c_str(String<cs...>) {
//   static constexpr const char str[sizeof...(cs) + 1] = {cs..., '\0'};
//   return str;
// }
//
// }  // namespace token
//
//// template <typename T>
//// static void* signature() {
////   static char id = 0;
////   return &id;
//// }
////
//// using signature_t = decltype(signature<void>());
//
// template <typename T, typename Key>
// struct Token {
//  using signature_t = void*;
//  using value_type = T;
//
//  Token(T value = {}) : value_(value) {}  // NOLINT
//
//  operator T() const { return value_; }  // NOLINT
//  static const char* key() { return token::c_str(Key{}); }
//
//  T& operator*() { return value_; }
//  T* operator->() { return &value_; }
//
// private:
//  T value_;
//};
//
// template <typename T>
// class Identifier {
// public:
//  constexpr explicit Identifier(const char* key) : key_(key) {}
//  const char* key_;
//};
//
// constexpr inline Identifier<int> batch_size{"batch_size"};
//
//}  // namespace mmdeploy
//
//#endif  // MMDEPLOY_SRC_TOKEN_TOKEN_H_
