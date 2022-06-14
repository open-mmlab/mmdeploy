// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_UTILS_FORMATTER_H_
#define MMDEPLOY_SRC_UTILS_FORMATTER_H_

#include <ostream>
#include <utility>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/types.h"
#include "spdlog/fmt/ostr.h"

#if FMT_VERSION >= 50000
#include "spdlog/fmt/bundled/ostream.h"
#include "spdlog/fmt/bundled/ranges.h"
#else
#include <type_traits>
#endif

namespace mmdeploy {

class Value;

MMDEPLOY_API std::string format_value(const Value& value);

inline std::string to_string(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBGR:
      return "BGR";
    case PixelFormat::kRGB:
      return "RGB";
    case PixelFormat::kGRAYSCALE:
      return "GRAYSCALE";
    case PixelFormat::kNV12:
      return "NV12";
    case PixelFormat::kNV21:
      return "NV21";
    case PixelFormat::kBGRA:
      return "BGRA";
    default:
      return "invalid_format_enum";
  }
}

inline std::string to_string(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return "FLOAT";
    case DataType::kHALF:
      return "HALF";
    case DataType::kINT8:
      return "INT8";
    case DataType::kINT32:
      return "INT32";
    case DataType::kINT64:
      return "INT64";
    default:
      return "invalid_data_type_enum";
  }
}

inline std::ostream& operator<<(std::ostream& os, PixelFormat format) {
  return os << to_string(format);
}

inline std::ostream& operator<<(std::ostream& os, DataType type) { return os << to_string(type); }

}  // namespace mmdeploy

namespace fmt {

#if FMT_VERSION >= 50000

// `Value` maybe an incomplete type at this point, making `operator<<` not usable
template <>
struct formatter<mmdeploy::Value> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  auto format(const mmdeploy::Value& value, Context& ctx) {
    return format_to(ctx.out(), "{}", mmdeploy::format_value(value));
  }
};

#else

inline void format_arg(BasicFormatter<char>& f, const char*, const mmdeploy::Value& d) {
  f.writer() << mmdeploy::format_value(d);
}

template <typename T>
auto format_arg(BasicFormatter<char>& f, const char*, const T& v)
    -> std::void_t<decltype(begin(v), end(v))> {
  f.writer() << "[";
  bool first = true;
  for (const auto& x : v) {
    f.writer() << (first ? "" : ", ") << fmt::format("{}", x);
    first = false;
  }
  f.writer() << "]";
}

template <class Tuple, size_t... Is>
void format_tuple_impl(BasicFormatter<char>& f, const Tuple& t, std::index_sequence<Is...>) {
  constexpr int last = sizeof...(Is) - 1;
  f.writer() << "(";
  ((f.writer() << fmt::format("{}", std::get<Is>(t)) << (Is != last ? ", " : "")), ...);
  f.writer() << ")";
}

template <typename... Ts>
void format_arg(BasicFormatter<char>& f, const char*, const std::tuple<Ts...>& t) {
  format_tuple_impl(f, t, std::index_sequence_for<Ts...>{});
}

#endif

}  // namespace fmt

#endif  // MMDEPLOY_SRC_UTILS_FORMATTER_H_
