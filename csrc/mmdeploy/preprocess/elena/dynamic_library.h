// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

namespace mmdeploy {
namespace elena {

class DynamicLibrary {
 public:
  DynamicLibrary(const char* name);

  void* Sym(const char* name);

  ~DynamicLibrary();

 private:
  void* handle_{};
};

}  // namespace elena
}  // namespace mmdeploy
