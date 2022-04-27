// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_UTILS_SOURCE_LOCATION_H_
#define MMDEPLOY_SRC_UTILS_SOURCE_LOCATION_H_

// clang-format off
#if __has_include(<source_location>) && (!_MSC_VER || __cplusplus >= 202002L)
  #include <source_location>
  #if __cpp_lib_source_location >= 201907L
    #define MMDEPLOY_HAS_SOURCE_LOCATION 1
    namespace mmdeploy {
    using SourceLocation = std::source_location;
    }
  #endif
#endif

#ifndef MMDEPLOY_HAS_SOURCE_LOCATION
  #if __has_include(<experimental/source_location>)
    #include <experimental/source_location>
    #if __cpp_lib_experimental_source_location >= 201505L
      #define MMDEPLOY_HAS_SOURCE_LOCATION 1
      namespace mmdeploy {
      using SourceLocation = std::experimental::source_location;
      }
    #endif
  #endif
#endif
// clang-format on

#ifndef MMDEPLOY_HAS_SOURCE_LOCATION
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
