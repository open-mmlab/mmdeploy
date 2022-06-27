#pragma once

#include <sstream>
#include <string>

#define DECLARE_PRIVATE_CONSTRUCT                       \
 private:                                               \
  static constexpr struct private_construct_t {         \
    explicit constexpr private_construct_t() = default; \
  } private_construct{};

#define DEFINE_PRIVATE_CONSTRUCT(Class) \
  constexpr struct Class::private_construct_t Class::private_construct;

namespace utils {
/// convert pointer to hexadecimal string.
inline std::string to_string(void* p) {
  std::ostringstream oss;
  oss << p;
  return oss.str();
}
}  // namespace utils
