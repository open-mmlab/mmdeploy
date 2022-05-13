// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_UTILS_FORMATTER_H_
#define MMDEPLOY_SRC_UTILS_FORMATTER_H_

#include <utility>

#include "core/logger.h"

#if FMT_VERSION >= 50000
#include "spdlog/fmt/bundled/ranges.h"
#include "spdlog/fmt/bundled/ostream.h"
#else
#include <type_traits>
#endif

namespace mmdeploy {

class Value;

MMDEPLOY_API std::string format_value(const Value& value);

}  // namespace mmdeploy

namespace fmt {

#if FMT_VERSION >= 50000

template <>
struct formatter<mmdeploy::Value> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  auto format(const mmdeploy::Value& value, Context& ctx) {
    return format_to(ctx.out(), "{}", mmdeploy::format_value(value));
  }
};

#else

inline void format_arg(BasicFormatter<char> &f, const char *, const mmdeploy::Value &d) {
  f.writer() << mmdeploy::format_value(d);
}

template <typename T, std::enable_if_t<std::is_enum<std::decay_t<T> >::value, bool> = true>
void format_arg(BasicFormatter<char> &f, const char *, const T &v) {
  f.writer() << (int)v;
}

template <typename T>
auto format_arg(BasicFormatter<char> &f, const char *, const T &v)
    -> std::void_t<decltype(begin(v), end(v))> {
  f.writer() << "[";
  bool first = true;
  for (const auto &x : v) {
    f.writer() << (first ? "" : ", ") << fmt::format("{}", x);
    first = false;
  }
  f.writer() << "]";
}

template <class Tuple, size_t... Is>
void format_tuple_impl(BasicFormatter<char> &f, const Tuple &t, std::index_sequence<Is...>) {
  constexpr int last = sizeof...(Is) - 1;
  f.writer() << "(";
  ((f.writer() << fmt::format("{}", std::get<Is>(t)) << (Is != last ? ", " : "")), ...);
  f.writer() << ")";
}

template <typename... Ts>
void format_arg(BasicFormatter<char> &f, const char *, const std::tuple<Ts...> &t) {
  format_tuple_impl(f, t, std::index_sequence_for<Ts...>{});
}

#endif

}  // namespace fmt

#endif  // MMDEPLOY_SRC_UTILS_FORMATTER_H_
