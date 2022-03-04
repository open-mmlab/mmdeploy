// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_UTILS_SOURCE_LOCATION_H_
#define MMDEPLOY_SRC_UTILS_SOURCE_LOCATION_H_

#if __has_include(<source_location>) && !_MSC_VER
#include <source_location>
namespace mmdeploy {
using SourceLocation = std::source_location;
}
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
namespace mmdeploy {
using SourceLocation = std::experimental::source_location;
}
#else
#include <cstdint>
namespace mmdeploy {
class SourceLocation {
 public:
  constexpr SourceLocation() noexcept = default;
  SourceLocation(const SourceLocation&) = default;
  SourceLocation(SourceLocation&&) noexcept = default;
  constexpr std::uint_least32_t line() const noexcept { return 0; };
  constexpr std::uint_least32_t column() const noexcept { return 0; }
  constexpr const char* file_name() const noexcept { return ""; }
  constexpr const char* function_name() const noexcept { return ""; }
  static constexpr SourceLocation current() noexcept { return {}; }
};
}  // namespace mmdeploy
#endif

#endif  // MMDEPLOY_SRC_UTILS_SOURCE_LOCATION_H_
