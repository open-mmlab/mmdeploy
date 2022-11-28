// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/status_code.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/source_location.h"

namespace mmdeploy {

void StatusDomain::_do_throw_exception(
    const SYSTEM_ERROR2_NAMESPACE::status_code<void> &code) const {
  assert(code.domain() == *this);
  const auto &c = static_cast<const StatusCode &>(code);  // NOLINT
  throw SYSTEM_ERROR2_NAMESPACE::status_error(c);
}

using string_ref = SYSTEM_ERROR2_NAMESPACE::status_code_domain::string_ref;
using atomic_refcounted_string_ref =
    SYSTEM_ERROR2_NAMESPACE::status_code_domain::atomic_refcounted_string_ref;

string_ref Status::message() const {
  std::string ret;
  try {
#if MMDEPLOY_STATUS_USE_SOURCE_LOCATION
#if MMDEPLOY_HAS_SOURCE_LOCATION
    ret = fmt::format("{} ({}) @ {}:{}", to_string(ec), (int32_t)ec, file, line);
#else
    ret = fmt::format("{} ({})", to_string(ec), (int32_t)ec);
#endif
#elif MMDEPLOY_STATUS_USE_STACKTRACE
    ret = fmt::format("{} ({}), stacktrace:\n{}", to_string(ec), (int32_t)ec, st.to_string());
#else
    ret = fmt::format("{} ({})", to_string(ec), (int32_t)ec);
#endif

  } catch (...) {
    return string_ref("Failed to retrieve message for status");
  }
  if (auto p = static_cast<char *>(malloc(ret.size() + 1))) {
    memcpy(p, ret.c_str(), ret.size() + 1);
    return atomic_refcounted_string_ref(p, ret.size());
  } else {
    return string_ref("Failed to allocate memory to store error string");
  }
}

}  // namespace mmdeploy
