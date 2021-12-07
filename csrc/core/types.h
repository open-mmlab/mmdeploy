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
  kBGRA
};

enum class DataType : int32_t {
  kFLOAT,
  kHALF,
  kINT8,
  kINT32,
  kINT64
};

// clang-format on

class NonCopyable {
 public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

}  // namespace mmdeploy

#endif  // !CORE_TYPES_H
