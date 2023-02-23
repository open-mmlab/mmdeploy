// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MPL_TYPE_TRAITS_H_
#define MMDEPLOY_SRC_CORE_MPL_TYPE_TRAITS_H_

#include <cstdint>
#include <type_traits>

namespace mmdeploy {

template <typename T>
struct uncvref {
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

template <typename T>
using uncvref_t = typename uncvref<T>::type;

template <class T>
struct is_cast_by_erasure : std::false_type {};

namespace traits {

using type_id_t = uint64_t;

template <class T>
struct TypeId {
  static constexpr type_id_t value = 0;
};

template <>
struct TypeId<void> {
  static constexpr auto value = static_cast<type_id_t>(-1);
};

// ! This only works when calling inside mmdeploy namespace
#define MMDEPLOY_REGISTER_TYPE_ID(type, id) \
  namespace traits {                        \
  template <>                               \
  struct TypeId<type> {                     \
    static constexpr type_id_t value = id;  \
  };                                        \
  }                                         \
  template <>                               \
  struct is_cast_by_erasure<type> : std::true_type {};
}  // namespace traits

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_MPL_TYPE_TRAITS_H_
