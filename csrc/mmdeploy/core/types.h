// Copyright (c) OpenMMLab. All rights reserved.

#ifndef CORE_TYPES_H
#define CORE_TYPES_H

#include <cstdint>

typedef int err_t;

namespace mmdeploy {

// clang-format off

enum class PixelFormat : int32_t {
  kBGR,
  kRGB,
  kGRAYSCALE,
  kNV12,
  kNV21,
  kBGRA,
  kCOUNT
};



enum class DataType : int32_t {
  kFLOAT,
  kHALF,
  kINT8,
  kINT32,
  kINT64,
  kCOUNT
};

// clang-format on

namespace pixel_formats {

constexpr auto kBGR = PixelFormat::kBGR;
constexpr auto kRGB = PixelFormat::kRGB;
constexpr auto kGRAY = PixelFormat::kGRAYSCALE;
constexpr auto kNV12 = PixelFormat::kNV12;
constexpr auto kNV21 = PixelFormat::kNV21;
constexpr auto kBGRA = PixelFormat::kBGRA;

}  // namespace pixel_formats

namespace data_types {

constexpr auto kFLOAT = DataType::kFLOAT;
constexpr auto kHALF = DataType::kHALF;
constexpr auto kINT8 = DataType::kINT8;
constexpr auto kINT32 = DataType::kINT32;
constexpr auto kINT64 = DataType::kINT64;

}  // namespace data_types

class NonCopyable {
 public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

class NonMovable {
 public:
  NonMovable() = default;
  NonMovable(const NonCopyable&) = delete;
  NonMovable& operator=(const NonCopyable&) = delete;
  NonMovable(NonMovable&&) noexcept = delete;
  NonMovable& operator=(NonMovable&&) noexcept = delete;
};

}  // namespace mmdeploy

#endif  // !CORE_TYPES_H
