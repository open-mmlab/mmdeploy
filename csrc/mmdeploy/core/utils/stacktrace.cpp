// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/stacktrace.h"

#if USE_BOOST_STACKTRACE

#define BOOST_STACKTRACE_USE_BACKTRACE
#include "boost/stacktrace.hpp"

namespace mmdeploy {

struct Stacktrace::Impl {
  boost::stacktrace::stacktrace st_;
};
Stacktrace::~Stacktrace() = default;
Stacktrace::Stacktrace(int)
    : impl_(new Impl{boost::stacktrace::stacktrace(1, static_cast<std::size_t>(-1))}) {}
Stacktrace::Stacktrace() noexcept = default;
Stacktrace::Stacktrace(const Stacktrace& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}
Stacktrace::Stacktrace(Stacktrace&& other) noexcept : impl_(std::move(other.impl_)) {}
Stacktrace& Stacktrace::operator=(Stacktrace&& other) noexcept {
  impl_ = std::move(other.impl_);
  return *this;
}
Stacktrace& Stacktrace::operator=(const Stacktrace& other) {
  impl_ = std::make_unique<Impl>(*other.impl_);
  return *this;
}
std::string Stacktrace::to_string() const {
  if (impl_) {
    return boost::stacktrace::to_string(impl_->st_);
  }
  return "";
}

}  // namespace mmdeploy

#else
#include <string>
namespace mmdeploy {

struct Stacktrace::Impl {};
Stacktrace::~Stacktrace() = default;
Stacktrace::Stacktrace(int) {}
Stacktrace::Stacktrace() noexcept = default;
Stacktrace::Stacktrace(const Stacktrace&) {}
Stacktrace::Stacktrace(Stacktrace&&) noexcept {}
Stacktrace& Stacktrace::operator=(Stacktrace&&) noexcept { return *this; }
Stacktrace& Stacktrace::operator=(const Stacktrace&) { return *this; }
std::string Stacktrace::to_string() const {
  return "the library is compiled with no stacktrace support";
}

}  // namespace mmdeploy

#endif
