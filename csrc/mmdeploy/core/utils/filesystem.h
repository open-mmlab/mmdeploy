// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CORE_UTILS_FILESYSTEM_H_
#define MMDEPLOY_CSRC_CORE_UTILS_FILESYSTEM_H_

#if __GNUC__ >= 8 || _MSC_VER || __clang_major__ >= 7
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#endif  // MMDEPLOY_CSRC_CORE_UTILS_FILESYSTEM_H_
