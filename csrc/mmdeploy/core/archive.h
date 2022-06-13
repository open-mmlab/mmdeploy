// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_ARCHIVE_H_
#define MMDEPLOY_SRC_CORE_ARCHIVE_H_

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/serialization.h"

namespace mmdeploy {

template <typename T, typename A>
using member_load_t = decltype(std::declval<T&>().load(std::declval<A&>()));

template <typename T, typename A>
using member_save_t = decltype(std::declval<T&>().save(std::declval<A&>()));

template <typename T, typename A>
using member_serialize_t = decltype(std::declval<T&>().serialize(std::declval<A&>()));

template <typename T, typename A>
using has_member_load = detail::is_detected<member_load_t, T, A>;

template <typename T, typename A>
using has_member_save = detail::is_detected<member_save_t, T, A>;

template <typename T, typename A>
using has_member_serialize = detail::is_detected<member_serialize_t, T, A>;

template <typename T, typename A>
using adl_load_t = decltype(adl_serializer<T>::load(std::declval<A&>(), std::declval<T&>()));

template <typename T, typename A>
using has_adl_load = detail::is_detected<adl_load_t, T, A>;

template <typename T, typename A>
using adl_save_t = decltype(adl_serializer<T>::save(std::declval<A&>(), std::declval<T&>()));

template <typename T, typename A>
using has_adl_save = detail::is_detected<adl_save_t, T, A>;

template <typename T, typename A>
using adl_serialize_t =
    decltype(adl_serializer<T>::serialize(std::declval<A&>(), std::declval<T&>()));

template <typename T, typename A>
using has_adl_serialize = detail::is_detected<adl_serialize_t, T, A>;

namespace detail {
// ADL bridge for archives
class ArchiveBase {};

}  // namespace detail

template <typename Archive>
class OutputArchive : public detail::ArchiveBase {
 public:
  template <typename... Args>
  void operator()(Args&&... args) {
    (dispatch(std::forward<Args>(args)), ...);
  }

 private:
  template <typename T>
  void dispatch(T&& v) {
    auto& archive = static_cast<Archive&>(*this);
    if constexpr (has_member_save<T, Archive>::value) {
      std::forward<T>(v).save(archive);
    } else if constexpr (has_member_serialize<T, Archive>::value) {
      std::forward<T>(v).serialize(archive);
    } else if constexpr (has_adl_save<T, Archive>::value) {
      adl_serializer<T>::save(archive, std::forward<T>(v));
    } else if constexpr (has_adl_serialize<T, Archive>::value) {
      adl_serializer<T>::serialize(archive, std::forward<T>(v));
    } else {
      archive.native(std::forward<T>(v));
    }
  }
};

template <typename Archive>
class InputArchive : public detail::ArchiveBase {
 public:
  template <typename... Args>
  void operator()(Args&&... args) {
    (dispatch(std::forward<Args>(args)), ...);
  }

 private:
  template <typename T>
  void dispatch(T&& v) {
    auto& archive = static_cast<Archive&>(*this);
    if constexpr (has_member_load<T, Archive>::value) {
      std::forward<T>(v).load(archive);
    } else if constexpr (has_member_serialize<T, Archive>::value) {
      std::forward<T>(v).serialize(archive);
    } else if constexpr (has_adl_load<T, Archive>::value) {
      adl_serializer<T>::load(archive, std::forward<T>(v));
    } else if constexpr (has_adl_serialize<T, Archive>::value) {
      adl_serializer<T>::serialize(archive, std::forward<T>(v));
    } else {
      archive.native(std::forward<T>(v));
    }
  }
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_ARCHIVE_H_
