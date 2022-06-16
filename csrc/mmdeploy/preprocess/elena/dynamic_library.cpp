// Copyright (c) OpenMMLab. All rights reserved.

#include "dynamic_library.h"

#include <stdexcept>
#include <string>

#include "mmdeploy/core/logger.h"
#ifndef _WIN32
#include <dlfcn.h>
#else
#endif

namespace mmdeploy {
namespace elena {

#ifndef _WIN32

// unix
DynamicLibrary::DynamicLibrary(const char* name) {
  handle_ = dlopen(name, RTLD_NOW);
  if (!handle_) {
    MMDEPLOY_ERROR("can't load lib: {}", name);
    throw std::runtime_error(std::string("can't load lib: ") + name);
  }
}

void* DynamicLibrary::Sym(const char* name) {
  void* ptr = dlsym(handle_, name);
  if (!ptr) {
    MMDEPLOY_ERROR("can't find sym: {}", name);
    throw std::runtime_error(std::string("can't find sym: ") + name);
  }
  return ptr;
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle_) {
    return;
  }
  dlclose(handle_);
}

#else
// TODO

#endif
}  // namespace elena
}  // namespace mmdeploy
