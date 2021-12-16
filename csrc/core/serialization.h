// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_SERIALIZATION_H_
#define MMDEPLOY_SRC_CORE_SERIALIZATION_H_

#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/status_code.h"
#include "mpl/detected.h"
#include "mpl/type_traits.h"

namespace mmdeploy {

#define _MMDEPLOY_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, \
                          N, ...)                                                                \
  N

#define _MMDEPLOY_ARCHIVE_1(x) MMDEPLOY_NVP(x)
#define _MMDEPLOY_ARCHIVE_2(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_1(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_3(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_2(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_4(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_3(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_5(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_4(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_6(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_5(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_7(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_6(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_8(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_7(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_9(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_8(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_10(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_9(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_11(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_10(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_12(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_11(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_13(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_12(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_14(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_13(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_15(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_14(__VA_ARGS__)
#define _MMDEPLOY_ARCHIVE_16(x, ...) MMDEPLOY_NVP(x), _MMDEPLOY_ARCHIVE_15(__VA_ARGS__)

#define _MMDEPLOY_ARCHIVE_DISPATCH(...)                                                   \
  _MMDEPLOY_NTH_ARG(__VA_ARGS__, _MMDEPLOY_ARCHIVE_16(__VA_ARGS__),                       \
                    _MMDEPLOY_ARCHIVE_15(__VA_ARGS__), _MMDEPLOY_ARCHIVE_14(__VA_ARGS__), \
                    _MMDEPLOY_ARCHIVE_13(__VA_ARGS__), _MMDEPLOY_ARCHIVE_12(__VA_ARGS__), \
                    _MMDEPLOY_ARCHIVE_11(__VA_ARGS__), _MMDEPLOY_ARCHIVE_10(__VA_ARGS__), \
                    _MMDEPLOY_ARCHIVE_9(__VA_ARGS__), _MMDEPLOY_ARCHIVE_8(__VA_ARGS__),   \
                    _MMDEPLOY_ARCHIVE_7(__VA_ARGS__), _MMDEPLOY_ARCHIVE_6(__VA_ARGS__),   \
                    _MMDEPLOY_ARCHIVE_5(__VA_ARGS__), _MMDEPLOY_ARCHIVE_4(__VA_ARGS__),   \
                    _MMDEPLOY_ARCHIVE_3(__VA_ARGS__), _MMDEPLOY_ARCHIVE_2(__VA_ARGS__),   \
                    _MMDEPLOY_ARCHIVE_1(__VA_ARGS__))

#define MMDEPLOY_ARCHIVE_NVP(archive, ...) archive(_MMDEPLOY_ARCHIVE_DISPATCH(__VA_ARGS__))

#define MMDEPLOY_ARCHIVE(archive, ...) archive(__VA_ARGS__)

#define MMDEPLOY_ARCHIVE_MEMBERS(...)           \
  template <typename Archive>                   \
  void serialize(Archive &archive) {            \
    MMDEPLOY_ARCHIVE_NVP(archive, __VA_ARGS__); \
  }

#define MMDEPLOY_NVP(var) \
  ::mmdeploy::NamedValue { std::forward_as_tuple(#var, var) }

template <typename NameT, typename ValueT>
class NamedValue {
 public:
  explicit NamedValue(std::tuple<NameT, ValueT> &&data) : data_(std::move(data)) {}
  template <typename Archive>
  void serialize(Archive &archive) {
    archive.named_value(std::forward<NameT>(std::get<0>(data_)),
                        std::forward<ValueT>(std::get<1>(data_)));
  }
  std::tuple<NameT, ValueT> &data() { return data_; }

 private:
  std::tuple<NameT, ValueT> data_;
};

template <typename T>
struct array_tag {
  explicit array_tag(std::size_t size) : size_(size) {}
  std::size_t size() const { return size_; }
  std::size_t size_;
};

template <typename T>
struct object_tag {};

template <typename T>
using mapped_type_t = typename T::mapped_type;

template <typename T>
using has_mapped_type = detail::is_detected<mapped_type_t, T>;

template <typename T>
using get_size_t = decltype(std::declval<T>().size());

template <typename T>
using has_size = detail::is_detected<get_size_t, T>;

template <typename T>
using reserve_t = decltype(std::declval<T>().reserve(std::size_t{0}));

template <typename T>
using has_reserve = detail::is_detected<reserve_t, T>;

namespace detail {

template <typename Archive, typename T, typename U = uncvref_t<T>,
          typename ValueType = typename U::value_type,
          std::enable_if_t<!std::is_same_v<U, std::string>, int> = 0>
auto save(Archive &archive, T &&iterable)
    -> std::void_t<decltype(iterable.begin(), iterable.end())> {
  if constexpr (has_size<T>::value) {
    archive.init(array_tag<ValueType>(iterable.size()));
  }
  for (auto &&x : iterable) {
    archive.item(std::forward<decltype(x)>(x));
  }
}

template <typename T0, typename T1>
class KeyValue {
 public:
  explicit KeyValue(std::tuple<T0, T1> &&data) : data_(std::move(data)) {}
  template <typename Archive>
  void serialize(Archive &archive) {
    archive.named_value("key", std::forward<T0>(std::get<0>(data_)));
    archive.named_value("value", std::forward<T1>(std::get<1>(data_)));
  }
  std::tuple<T0, T1> &data() { return data_; }

 private:
  std::tuple<T0, T1> data_;
};

template <typename Archive, typename T, typename U = uncvref_t<T>,
          typename KeyType = typename U::key_type, typename MappedType = typename U::mapped_type,
          std::enable_if_t<!std::is_constructible_v<std::string, KeyType>, int> = 0>
auto save(Archive &archive, T &object) -> std::void_t<decltype(object.begin(), object.end())> {
  if constexpr (has_size<T>::value) {
    // TODO: provide meaningful type info
    archive.init(array_tag<void>(object.size()));
  }
  for (auto &&[k, v] : object) {
    archive.item(KeyValue{
        std::forward_as_tuple(std::forward<decltype(k)>(k), std::forward<decltype(v)>(v))});
  }
}

template <typename Archive, typename T, typename U = uncvref_t<T>,
          typename KeyType = typename U::key_type, typename MappedType = typename U::mapped_type,
          std::enable_if_t<std::is_constructible_v<std::string, KeyType>, int> = 0>
auto save(Archive &archive, T &object) -> std::void_t<decltype(object.begin(), object.end())> {
  if constexpr (has_size<T>::value) {
    archive.init(object_tag<MappedType>());
  }
  for (auto &&[k, v] : object) {
    archive.named_value(std::forward<decltype(k)>(k), std::forward<decltype(v)>(v));
  }
}

template <typename Archive, typename T, std::size_t... Is>
void save_tuple_impl(Archive &archive, T &&tuple, std::index_sequence<Is...>) {
  (archive.item(std::get<Is>(std::forward<T>(tuple))), ...);
}

template <typename Archive, typename... Ts>
void save(Archive &archive, const std::tuple<Ts...> &tuple) {
  save_tuple_impl(archive, tuple, std::index_sequence_for<Ts...>{});
}

template <typename Archive, typename T, size_t... Is>
void load_tuple_impl(Archive &archive, T &tuple, std::index_sequence<Is...>) {
  (archive.item(std::get<Is>(tuple)), ...);
}

template <typename Archive, typename T, std::size_t N>
void save(Archive &archive, T (&v)[N]) {
  archive.init(array_tag<T>(N));
  for (std::size_t i = 0; i < N; ++i) {
    archive.item(v[i]);
  }
}

template <typename Archive, typename... Ts>
void load(Archive &archive, std::tuple<Ts...> &tuple) {
  std::size_t size{};
  archive.init(size);
  if (size != sizeof...(Ts)) {
    throw_exception(eShapeMismatch);
  }
  load_tuple_impl(archive, tuple, std::index_sequence_for<Ts...>{});
}

template <typename Archive, typename T, typename U = uncvref_t<T>,
          typename ValueType = typename U::value_type,
          std::enable_if_t<!std::is_same_v<U, std::string>, int> = 0>
auto load(Archive &&archive, T &&vec) -> std::void_t<decltype(vec.push_back(ValueType{}))> {
  std::size_t size{};
  archive.init(size);
  vec.clear();
  for (std::size_t i = 0; i < size; ++i) {
    ValueType v{};
    archive.item(v);
    vec.push_back(std::move(v));
  }
}

template <typename Archive, typename T, std::size_t N>
void load(Archive &archive, std::array<T, N> &v) {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    archive.item(v[i]);
  }
}

template <typename Archive, typename T, std::size_t N>
void load(Archive &archive, T (&v)[N]) {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    archive.item(v[i]);
  }
}

template <typename Archive, typename T, typename U = uncvref_t<T>,
          typename ValueType = typename U::value_type,
          std::enable_if_t<std::conjunction_v<std::is_default_constructible<ValueType>,
                                              std::negation<has_mapped_type<U>>>,
                           int> = 0>
auto load(Archive &&archive, T &&set)
    -> std::void_t<decltype(set.insert(std::declval<ValueType>()))> {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    ValueType v{};
    archive.item(v);
    set.insert(std::move(v));
  }
}

template <
    typename Archive, typename T, typename U = uncvref_t<T>,
    typename KeyType = typename U::key_type, typename MappedType = typename U::mapped_type,
    std::enable_if_t<std::conjunction_v<std::negation<std::is_constructible<KeyType, std::string>>,
                                        std::is_default_constructible<KeyType>,
                                        std::is_default_constructible<MappedType>>,
                     int> = 0>
void load(Archive &&archive, T &&object) {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    KeyType key;
    MappedType mapped;
    archive.item(KeyValue{std::tie(key, mapped)});
    object.insert({std::move(key), std::move(mapped)});
  };
}

template <typename Archive, typename T, typename U = uncvref_t<T>,
          typename KeyType = typename U::key_type, typename MappedType = typename U::mapped_type,
          std::enable_if_t<std::conjunction_v<std::is_constructible<KeyType, std::string>,
                                              std::is_default_constructible<MappedType>>,
                           int> = 0>
void load(Archive &&archive, T &&object) {
  std::size_t size{};
  archive.init(size);
  for (std::size_t i = 0; i < size; ++i) {
    std::string name;
    MappedType value{};
    archive.named_value(name, value);
    object.insert({std::move(name), std::move(value)});
  }
}

template <typename Archive, typename T>
using save_t = decltype(save(std::declval<Archive>(), std::declval<T>()));

struct save_fn {
  template <typename Archive, typename T>
  auto operator()(Archive &&a, T &&v) const -> save_t<Archive, T> {
    return save(std::forward<Archive>(a), std::forward<T>(v));
  }
};

template <typename Archive, typename T>
using load_t = decltype(load(std::declval<Archive>(), std::declval<T>()));

struct load_fn {
  template <typename Archive, typename T>
  auto operator()(Archive &&a, T &&v) const -> load_t<Archive, T> {
    return load(std::forward<Archive>(a), std::forward<T>(v));
  }
};

template <typename Archive, typename T>
using serialize_t = decltype(serialize(std::declval<Archive>(), std::declval<T>()));

struct serialize_fn {
  template <typename Archive, typename T>
  auto operator()(Archive &&a, T &&v) const -> serialize_t<Archive, T> {
    return serialize(std::forward<Archive>(a), std::forward<T>(v));
  }
};

}  // namespace detail

namespace {

constexpr inline detail::save_fn save{};
constexpr inline detail::load_fn load{};
constexpr inline detail::serialize_fn serialize{};

}  // namespace

template <typename T = void, typename SFINAE = void>
struct adl_serializer;

template <typename, typename>
struct adl_serializer {
  template <typename Archive, typename T>
  static auto save(Archive &&a, T &&v) -> detail::save_t<Archive, T> {
    ::mmdeploy::save(std::forward<Archive>(a), std::forward<T>(v));
  }
  template <typename Archive, typename T>
  static auto load(Archive &&a, T &&v) -> detail::load_t<Archive, T> {
    ::mmdeploy::load(std::forward<Archive>(a), std::forward<T>(v));
  }
  template <typename Archive, typename T>
  static auto serialize(Archive &&a, T &&v) -> detail::serialize_t<Archive, T> {
    ::mmdeploy::serialize(std::forward<Archive>(a), std::forward<T>(v));
  }
};

};  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_SERIALIZATION_H_
