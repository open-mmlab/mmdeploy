// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_STATUS_CODE_H_
#define MMDEPLOY_SRC_CORE_STATUS_CODE_H_

#include <system_error>

#include "mmdeploy/core/macro.h"
#include "outcome-experimental.hpp"
#if MMDEPLOY_STATUS_USE_SOURCE_LOCATION
#include "mmdeploy/core/utils/source_location.h"
#elif MMDEPLOY_STATUS_USE_STACKTRACE
#include "mmdeploy/core/utils/stacktrace.h"
#endif

namespace mmdeploy {

// clang-format off

enum class ErrorCode: int32_t {
  eSuccess         = 0,
  eInvalidArgument = 1,
  eNotSupported    = 2,
  eOutOfRange      = 3,
  eOutOfMemory     = 4,
  eFileNotExist    = 5,
  eFail            = 6,
  eShapeMismatch   = 7,
  eEntryNotFound   = 8,
  eNotReady        = 9,
  eUnknown         = -1,
};

// clang-format on

#define USING_ERROR_CODE(code) constexpr inline const auto code = ErrorCode::code  // NOLINT

// note that eSuccess is not brought to the outer namespace on purpose
USING_ERROR_CODE(eInvalidArgument);
USING_ERROR_CODE(eNotSupported);
USING_ERROR_CODE(eOutOfRange);
USING_ERROR_CODE(eOutOfMemory);
USING_ERROR_CODE(eFileNotExist);
USING_ERROR_CODE(eFail);
USING_ERROR_CODE(eShapeMismatch);
USING_ERROR_CODE(eEntryNotFound);
USING_ERROR_CODE(eNotReady);
USING_ERROR_CODE(eUnknown);

inline const char *to_string(ErrorCode code) {
  switch (code) {
    case ErrorCode::eSuccess:
      return "success";
    case ErrorCode::eInvalidArgument:
      return "invalid argument";
    case ErrorCode::eNotSupported:
      return "not supported";
    case ErrorCode::eOutOfRange:
      return "out of range";
    case ErrorCode::eOutOfMemory:
      return "out of memory";
    case ErrorCode::eFileNotExist:
      return "file not exist";
    case ErrorCode::eShapeMismatch:
      return "shape mismatch";
    case ErrorCode::eEntryNotFound:
      return "entry not found";
    case ErrorCode::eNotReady:
      return "not ready";
    default:
      return "unknown";
  }
}

struct MMDEPLOY_API Status {
  ErrorCode ec{};
  Status() = default;
  SYSTEM_ERROR2_NAMESPACE::status_code_domain::string_ref message() const;
  bool operator==(const ErrorCode &b) const noexcept { return ec == b; }

#if MMDEPLOY_STATUS_USE_SOURCE_LOCATION
  const char *file{""};
  int line{};
  explicit Status(ErrorCode _ec, SourceLocation location = SourceLocation::current())
      : ec(_ec), file(location.file_name()), line(static_cast<int>(location.line())) {}
#elif MMDEPLOY_STATUS_USE_STACKTRACE
  Stacktrace st;
  explicit Status(ErrorCode _ec, Stacktrace _st = Stacktrace(0)) : ec(_ec), st(std::move(_st)) {}
#else
  explicit Status(ErrorCode _ec) : ec(_ec) {}
#endif
};

class StatusDomain;

using StatusCode = SYSTEM_ERROR2_NAMESPACE::status_code<StatusDomain>;

class MMDEPLOY_API StatusDomain : public SYSTEM_ERROR2_NAMESPACE::status_code_domain {
  using _base = status_code_domain;

 public:
  using value_type = Status;

  constexpr explicit StatusDomain(typename _base::unique_id_type id = 0x3584b6716049efb4) noexcept
      : _base(id) {}

  StatusDomain(const StatusDomain &) = default;
  StatusDomain(StatusDomain &&) = default;
  StatusDomain &operator=(const StatusDomain &) = default;
  StatusDomain &operator=(StatusDomain &&) = default;
  ~StatusDomain() = default;

  static inline constexpr const StatusDomain &get();

  string_ref name() const noexcept override {
    static string_ref v("mmdeploy");
    return v;
  }

  // clang-format off
  bool _do_failure(const SYSTEM_ERROR2_NAMESPACE::status_code<void> &code) const noexcept override {
    assert(code.domain() == *this);
    auto &c = static_cast<const StatusCode &>(code);  // NOLINT
    return c.value().ec != ErrorCode::eSuccess;
  }
  bool _do_equivalent(
      const SYSTEM_ERROR2_NAMESPACE::status_code<void> &code1,
      const SYSTEM_ERROR2_NAMESPACE::status_code<void> &code2) const noexcept override {
    assert(code1.domain() == *this);
    if (code1.domain() == *this && code2.domain() == *this) {
      auto &c1 = static_cast<const StatusCode &>(code1);  // NOLINT
      auto &c2 = static_cast<const StatusCode &>(code2);  // NOLINT
      return c1.value().ec == c2.value().ec;
    }
    return false;
  }
  SYSTEM_ERROR2_NAMESPACE::generic_code _generic_code(
      const SYSTEM_ERROR2_NAMESPACE::status_code<void> &code) const noexcept override {
    assert(code.domain() == *this);
    return SYSTEM_ERROR2_NAMESPACE::errc::unknown;
  }
  string_ref _do_message(
      const SYSTEM_ERROR2_NAMESPACE::status_code<void> &code) const noexcept override {
    assert(code.domain() == *this);
    auto &c = static_cast<const StatusCode &>(code);  // NOLINT
    return c.value().message();
  }
  // clang-format on
  void _do_throw_exception(const SYSTEM_ERROR2_NAMESPACE::status_code<void> &code) const override;
};

constexpr inline StatusDomain status_domain;
inline constexpr const StatusDomain &StatusDomain::get() { return status_domain; }

inline StatusCode make_status_code(StatusCode::value_type v) {
  return StatusCode(SYSTEM_ERROR2_NAMESPACE::in_place, static_cast<StatusCode::value_type &&>(v));
}

using OUTCOME_V2_NAMESPACE::failure;
using OUTCOME_V2_NAMESPACE::in_place_type;
using OUTCOME_V2_NAMESPACE::success;

inline bool operator==(const StatusCode &sc, ErrorCode ec) noexcept { return sc.value().ec == ec; }
inline bool operator==(ErrorCode ec, const StatusCode &sc) noexcept { return sc.value().ec == ec; }

using Error = SYSTEM_ERROR2_NAMESPACE::errored_status_code<StatusDomain>;

using Exception = SYSTEM_ERROR2_NAMESPACE::status_error<StatusDomain>;

template <typename T>
using Result = OUTCOME_V2_NAMESPACE::experimental::status_result<T, Error>;

#if MMDEPLOY_STATUS_USE_SOURCE_LOCATION
[[noreturn]] inline void throw_exception(ErrorCode ec,
                                         SourceLocation location = SourceLocation::current()) {
  Error(Status(ec, location)).throw_exception();
}
#elif MMDEPLOY_STATUS_USE_STACKTRACE
[[noreturn]] inline void throw_exception(ErrorCode ec, Stacktrace stacktrace = Stacktrace(0)) {
  Error(Status(ec, std::move(stacktrace))).throw_exception();
}
#else
[[noreturn]] inline void throw_exception(const ErrorCode ec) {
  Error(Status(ec)).throw_exception();
}
#endif

template <typename T>
inline constexpr bool is_result_v = OUTCOME_V2_NAMESPACE::is_basic_result_v<T>;

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_STATUS_CODE_H_
