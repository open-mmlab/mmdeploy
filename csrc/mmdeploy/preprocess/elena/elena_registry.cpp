// Copyright (c) OpenMMLab. All rights reserved.

#include "elena_registry.h"

#include "mmdeploy/core/logger.h"

namespace mmdeploy {
namespace elena {

FuseKernel& FuseKernel::Get() {
  static FuseKernel fuse_kernel;
  return fuse_kernel;
}

FuseFunc FuseKernel::GetFunc(const std::string& name) {
  if (entries_.count(name)) {
    return entries_[name];
  }
  return nullptr;
}

int FuseKernel::Register(const std::string& name, FuseFunc func) {
  if (entries_.count(name)) {
    return -1;
  }
  MMDEPLOY_DEBUG("Register fuse kernel: '{}'", name);
  entries_.emplace(name, func);
  return 0;
}

}  // namespace elena
}  // namespace mmdeploy
