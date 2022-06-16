// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <mutex>
#include <string>

#include "mmdeploy/core/value.h"

namespace mmdeploy {
namespace elena {

class Compiler {
 public:
  Compiler(const Compiler&) = delete;
  Compiler& operator=(const Compiler&) = delete;
  bool Compile(const Value& input, const std::string& platform_name, std::string& lib_name);
  static Compiler& Instance();

 private:
  Compiler();
  std::mutex mutex_;
  std::string folder_;
};

}  // namespace elena
}  // namespace mmdeploy