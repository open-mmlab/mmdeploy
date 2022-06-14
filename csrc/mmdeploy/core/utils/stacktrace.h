// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_STACKTRACE_H_
#define MMDEPLOY_SRC_CORE_STACKTRACE_H_

#include <memory>
#include <string>

namespace mmdeploy {

class Stacktrace {
 public:
  ~Stacktrace();
  Stacktrace() noexcept;
  explicit Stacktrace(int);
  Stacktrace& operator=(const Stacktrace&);
  Stacktrace& operator=(Stacktrace&& other) noexcept;
  Stacktrace(const Stacktrace&);
  Stacktrace(Stacktrace&&) noexcept;
  std::string to_string() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_STACKTRACE_H_
