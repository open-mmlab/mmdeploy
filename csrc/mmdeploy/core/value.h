// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TYPES_VALUE_H_
#define MMDEPLOY_TYPES_VALUE_H_

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mpl/priority_tag.h"
#include "mmdeploy/core/mpl/static_any.h"
#include "mmdeploy/core/mpl/type_traits.h"
#include "mmdeploy/core/status_code.h"

namespace mmdeploy {

enum class ValueType : int {
  kNull = 0,
  kBool,
  kInt,
  kUInt,
  kFloat,
  kString,
  kBinary,
  kArray,
  kObject,
  kPointer,
  kDynamic,
  kAny,
};

class Value;

#if __GNUC__ >= 8
using Byte = std::byte;
#else
enum class Byte : unsigned char {};
#endif

namespace detail {
class ValueRef;
}

template <typename T>
class ValueIterator {
 public:
  using value_type = Value;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::bidirectional_iterator_tag;
  using object_iterator_t = typename T::Object::iterator;
  using array_iterator_t = typename T::Array::iterator;
  ValueIterator() = default;
  ValueIterator(T* value, object_iterator_t iter) : value_(value), object_iter_(iter) {}
  ValueIterator(T* value, array_iterator_t iter) : value_(value), array_iter_(iter) {}
  ValueIterator& operator++() {
    if (value_->is_array()) {
      ++array_iter_;
    } else {
      ++object_iter_;
    }
    return *this;
  }
  ValueIterator operator++(int) {
    auto it = *this;
    ++(*this);
    return it;
  }
  T& operator*() {
    if (value_->is_array()) {
      return *array_iter_;
    } else {
      return object_iter_->second;
    }
  }
  const T& operator*() const {
    if (value_->is_array()) {
      return *array_iter_;
    } else {
      return object_iter_->second;
    }
  }
  T* operator->() {
    if (value_->is_array()) {
      return &(*array_iter_);
    } else {
      return &object_iter_->second;
    }
  }
  const T* operator->() const {
    if (value_->is_array()) {
      return &(*array_iter_);
    } else {
      return &object_iter_->second;
    }
  }
  const std::string& key() {
    if (value_->is_object()) {
      return object_iter_->first;
    }
    throw_exception(eInvalidArgument);
  }
  bool operator==(const ValueIterator& other) const {
    return value_ == other.value_ && object_iter_ == other.object_iter_ &&
           array_iter_ == other.array_iter_;
  }
  bool operator!=(const ValueIterator& other) const { return !(*this == other); }

 private:
  T* value_{};
  object_iterator_t object_iter_{};
  array_iterator_t array_iter_{};
};

class Dynamic;

class Value;

template <class T>
struct EraseType {
  T value;
};

template <class T>
struct ArchiveType {
  T value;
};

template <class T>
EraseType<T&&> cast_by_erasure(T&& v) {
  return {std::forward<T>(v)};
}

template <class T>
ArchiveType<T&&> cast_by_archive(T&& v) {
  return {std::forward<T>(v)};
}

template <class T>
struct is_cast_by_erasure : std::false_type {};

class Device;
class Buffer;
class Stream;
class Event;
class Model;
class Tensor;
class Mat;

template <>
struct is_cast_by_erasure<Device> : std::true_type {};
template <>
struct is_cast_by_erasure<Buffer> : std::true_type {};
template <>
struct is_cast_by_erasure<Stream> : std::true_type {};
template <>
struct is_cast_by_erasure<Event> : std::true_type {};
template <>
struct is_cast_by_erasure<Model> : std::true_type {};
template <>
struct is_cast_by_erasure<Tensor> : std::true_type {};
template <>
struct is_cast_by_erasure<Mat> : std::true_type {};

MMDEPLOY_REGISTER_TYPE_ID(Device, 1);
MMDEPLOY_REGISTER_TYPE_ID(Buffer, 2);
MMDEPLOY_REGISTER_TYPE_ID(Stream, 3);
MMDEPLOY_REGISTER_TYPE_ID(Event, 4);
MMDEPLOY_REGISTER_TYPE_ID(Model, 5);
MMDEPLOY_REGISTER_TYPE_ID(Tensor, 6);
MMDEPLOY_REGISTER_TYPE_ID(Mat, 7);

template <typename T>
struct is_value : std::is_same<T, Value> {};

template <typename T>
inline constexpr bool is_value_v = is_value<T>::value;

namespace detail {
template <typename T>
struct is_pointer_to_const : std::false_type {};
template <typename T>
struct is_pointer_to_const<const T*> : std::true_type {};
template <typename T>
struct is_const_reference : std::false_type {};
template <typename T>
struct is_const_reference<const T&> : std::true_type {};
}  // namespace detail

class Value {
 public:
  using value_type = Value;
  using reference = value_type&;
  using const_reference = const value_type&;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = ValueIterator<Value>;
  using const_iterator = ValueIterator<const Value>;

  using Type = ValueType;

  using Boolean = bool;
  using Integer = int64_t;
  using Unsigned = uint64_t;
  using Float = double;
  using String = std::string;
  using Binary = std::vector<Byte>;
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value>;
  using Pointer = std::shared_ptr<Value>;
  using Dynamic = ::mmdeploy::Dynamic;
  using Any = ::mmdeploy::StaticAny;
  using ValueRef = detail::ValueRef;

  static constexpr const auto kNull = ValueType::kNull;
  static constexpr const auto kBool = ValueType::kBool;
  static constexpr const auto kInt = ValueType::kInt;
  static constexpr const auto kUInt = ValueType::kUInt;
  static constexpr const auto kFloat = ValueType::kFloat;
  static constexpr const auto kString = ValueType::kString;
  static constexpr const auto kBinary = ValueType::kBinary;
  static constexpr const auto kArray = ValueType::kArray;
  static constexpr const auto kObject = ValueType::kObject;
  static constexpr const auto kPointer = ValueType::kPointer;
  static constexpr const auto kDynamic = ValueType::kDynamic;
  static constexpr const auto kAny = ValueType::kAny;

  Value(const ValueType v) : type_(v), data_(v) {}

  Value(std::nullptr_t = nullptr) noexcept : Value(ValueType::kNull) {}

  template <typename T, std::enable_if_t<std::is_same_v<T, ValueRef>, int> = 0>
  Value(const T& ref) : Value(ref.moved_or_copied()) {}

  Value(const Value& other) : type_(other.type_) {
    switch (type_) {
      case ValueType::kNull:
        break;
      case ValueType::kBool:
        data_ = other.data_.boolean;
        break;
      case ValueType::kInt:
        data_ = other.data_.number_integer;
        break;
      case ValueType::kUInt:
        data_ = other.data_.number_unsigned;
        break;
      case ValueType::kFloat:
        data_ = other.data_.number_float;
        break;
      case ValueType::kString:
        data_ = *other.data_.string;
        break;
      case ValueType::kBinary:
        data_ = *other.data_.binary;
        break;
      case ValueType::kArray:
        data_ = *other.data_.array;
        break;
      case ValueType::kObject:
        data_ = *other.data_.object;
        break;
      case ValueType::kPointer:
        data_ = *other.data_.pointer;
        break;
      case ValueType::kAny:
        data_.any = create<Any>(*other.data_.any);
        break;
      default:
        throw_exception(eInvalidArgument);
    }
  }

  template <class T, std::enable_if_t<std::is_same<std::decay_t<T>, bool>::value, bool> = true>
  Value(T&& value) : type_(kBool), data_(Boolean{value}) {}

  Value(int8_t value) : type_(kInt), data_(Integer{value}) {}
  Value(int16_t value) : type_(kInt), data_(Integer{value}) {}
  Value(int32_t value) : type_(kInt), data_(Integer{value}) {}
  Value(int64_t value) : type_(kInt), data_(Integer{value}) {}
  Value(uint8_t value) : type_(kUInt), data_(Unsigned{value}) {}
  Value(uint16_t value) : type_(kUInt), data_(Unsigned{value}) {}
  Value(uint32_t value) : type_(kUInt), data_(Unsigned{value}) {}
  Value(uint64_t value) : type_(kUInt), data_(Unsigned{value}) {}
  Value(float value) : type_(kFloat), data_(Float{value}) {}
  Value(double value) : type_(kFloat), data_(Float{value}) {}
  Value(Binary value) : type_(kBinary), data_(std::move(value)) {}
  Value(Array value) : type_(kArray), data_(std::move(value)) {}
  Value(Object value) : type_(kObject), data_(std::move(value)) {}
  Value(Pointer value) : type_(kPointer), data_(std::move(value)) {}

  template <class T, std::enable_if_t<std::is_constructible<String, T>::value, bool> = true>
  Value(T&& value) : type_(kString), data_(String{std::forward<T>(value)}) {}

  template <typename T, std::enable_if_t<is_cast_by_erasure<std::decay_t<T>>::value, bool> = true>
  Value(T&& value) : Value(cast_by_erasure(std::forward<T>(value))) {}

  template <typename T>
  Value(EraseType<T>&& value) : type_(Type::kAny) {
    data_.any = create<Any>(std::forward<T>(value.value));
  }

  Value(std::initializer_list<ValueRef> init, bool type_deduction = true,
        Type manual_type = Type::kArray);

  Value(Value&& other) noexcept : type_(other.type_), data_(other.data_) {
    other.type_ = ValueType::kNull;
    other.data_ = {};
  }

  // copy-and-swap
  Value& operator=(Value other) noexcept {
    using std::swap;
    swap(type_, other.type_);
    swap(data_, other.data_);
    return *this;
  }

  ~Value() { data_.destroy(type_); }

  operator Type() const noexcept { return type(); }
  Type type() const noexcept { return _unwrap().type_; }
  bool is_null() const noexcept { return _unwrap()._is_null(); }
  bool is_array() const noexcept { return _unwrap()._is_array(); }
  bool is_object() const noexcept { return _unwrap()._is_object(); }
  template <typename T = void>
  bool is_any() const noexcept {
    return _unwrap()._is_any<T>();
  }
  bool is_boolean() const noexcept { return _unwrap()._is_boolean(); }
  bool is_string() const noexcept { return _unwrap()._is_string(); }
  bool is_binary() const noexcept { return _unwrap()._is_binary(); }
  bool is_number() const noexcept { return _unwrap()._is_number(); }
  bool is_number_integer() const noexcept { return _unwrap()._is_number_integer(); }
  bool is_number_unsigned() const noexcept { return _unwrap()._is_number_unsigned(); }
  bool is_number_float() const noexcept { return _unwrap()._is_number_float(); }
  bool is_pointer() const noexcept { return _is_pointer(); }
  size_t size() const noexcept { return _unwrap()._size(); }
  bool empty() const noexcept { return _unwrap()._empty(); }

 private:
  constexpr Type _type() const noexcept { return type_; }

  constexpr bool _is_null() const noexcept { return type_ == Type::kNull; }
  constexpr bool _is_array() const noexcept { return type_ == Type::kArray; }
  constexpr bool _is_object() const noexcept { return type_ == Type::kObject; }

  template <typename T = void>
  constexpr bool _is_any() const noexcept {
    if (type_ != Type::kAny) {
      return false;
    }
    if constexpr (std::is_void_v<T>) {
      return true;
    } else {
      return traits::TypeId<T>::value == data_.any->type();
    }
  }

  constexpr bool _is_boolean() const noexcept { return type_ == Type::kBool; }
  constexpr bool _is_string() const noexcept { return type_ == Type::kString; }
  constexpr bool _is_binary() const noexcept { return type_ == Type::kBinary; }
  constexpr bool _is_number() const noexcept { return _is_number_integer() || _is_number_float(); }

  constexpr bool _is_number_integer() const noexcept {
    return type_ == Type::kInt || type_ == Type::kUInt;
  }

  constexpr bool _is_number_unsigned() const noexcept { return type_ == Type::kUInt; }
  constexpr bool _is_number_float() const noexcept { return type_ == Type::kFloat; }
  constexpr bool _is_pointer() const noexcept { return type_ == Type::kPointer; }

  size_t _size() const noexcept {
    switch (_type()) {
      case ValueType::kNull:
        return 0;
      case ValueType::kArray:
        return data_.array->size();
      case ValueType::kObject:
        return data_.object->size();
      default:
        return 1;
    }
  }

  bool _empty() const noexcept {
    switch (_type()) {
      case Type::kNull:
        return true;
      case Type::kArray:
        return data_.array->empty();
      case Type::kObject:
        return data_.object->empty();
      default:
        return false;
    }
  }

 private:
  Boolean* get_impl_ptr(Boolean*) noexcept { return _is_boolean() ? &data_.boolean : nullptr; }
  const Boolean* get_impl_ptr(const Boolean*) const noexcept {
    return _is_boolean() ? &data_.boolean : nullptr;
  }
  Integer* get_impl_ptr(Integer*) noexcept {
    return _is_number_integer() ? &data_.number_integer : nullptr;
  }
  const Integer* get_impl_ptr(const Integer*) const noexcept {
    return _is_number_integer() ? &data_.number_integer : nullptr;
  }
  Unsigned* get_impl_ptr(Unsigned*) noexcept {
    return _is_number_unsigned() ? &data_.number_unsigned : nullptr;
  }
  const Unsigned* get_impl_ptr(const Unsigned*) const noexcept {
    return _is_number_unsigned() ? &data_.number_unsigned : nullptr;
  }
  Float* get_impl_ptr(Float*) noexcept {
    return _is_number_float() ? &data_.number_float : nullptr;
  }
  const Float* get_impl_ptr(const Float*) const noexcept {
    return _is_number_float() ? &data_.number_float : nullptr;
  }
  String* get_impl_ptr(String*) noexcept { return _is_string() ? data_.string : nullptr; }
  const String* get_impl_ptr(const String*) const noexcept {
    return _is_string() ? data_.string : nullptr;
  }
  Binary* get_impl_ptr(Binary*) noexcept { return _is_binary() ? data_.binary : nullptr; }
  const Binary* get_impl_ptr(const Binary*) const noexcept {
    return _is_binary() ? data_.binary : nullptr;
  }
  Array* get_impl_ptr(Array*) noexcept { return _is_array() ? data_.array : nullptr; }
  const Array* get_impl_ptr(const Array*) const noexcept {
    return _is_array() ? data_.array : nullptr;
  }
  Object* get_impl_ptr(Object*) noexcept { return _is_object() ? data_.object : nullptr; }
  const Object* get_impl_ptr(const Object*) const noexcept {
    return _is_object() ? data_.object : nullptr;
  }
  Pointer* get_impl_ptr(Pointer*) noexcept { return _is_pointer() ? data_.pointer : nullptr; }
  const Pointer* get_impl_ptr(const Pointer*) const noexcept {
    return _is_pointer() ? data_.pointer : nullptr;
  }
  Any* get_impl_ptr(Any*) noexcept { return _is_any() ? data_.any : nullptr; }
  const Any* get_impl_ptr(const Any*) const noexcept { return _is_any() ? data_.any : nullptr; }

  template <typename T>
  T* get_erased_ptr(EraseType<T>*) noexcept {
    return _is_any() ? static_any_cast<T>(data_.any) : nullptr;
  }
  template <typename T>
  const T* get_erased_ptr(const EraseType<T>*) const noexcept {
    return _is_any() ? static_any_cast<T>(const_cast<const Any*>(data_.any)) : nullptr;
  }

  template <typename T, typename This>
  static auto get_ref_impl(This& obj)
      -> decltype((*obj.template get_ptr<std::add_pointer_t<T>>())) {
    auto p = obj.template get_ptr<std::add_pointer_t<T>>();
    if (p) {
      return *p;
    }
    throw_exception(eInvalidArgument);
  }

  template <typename T, std::enable_if_t<std::is_pointer<T>::value, bool> = true>
  auto _get_ptr() noexcept -> decltype(std::declval<Value&>().get_impl_ptr(std::declval<T>())) {
    return get_impl_ptr(static_cast<T>(nullptr));
  }

  template <typename T, std::enable_if_t<detail::is_pointer_to_const<T>::value, bool> = true>
  auto _get_ptr() const noexcept
      -> decltype(std::declval<const Value&>().get_impl_ptr(std::declval<T>())) {
    return get_impl_ptr(static_cast<T>(nullptr));
  }

  template <typename T, std::enable_if_t<std::is_pointer<T>::value, bool> = true>
  auto _get_ptr() noexcept -> decltype(std::declval<Value&>().get_erased_ptr(std::declval<T>())) {
    return get_erased_ptr(static_cast<T>(nullptr));
  }

  template <typename T, std::enable_if_t<detail::is_pointer_to_const<T>::value, bool> = true>
  auto _get_ptr() const noexcept
      -> decltype(std::declval<const Value&>().get_erased_ptr(std::declval<T>())) {
    return get_erased_ptr(static_cast<T>(nullptr));
  }

  // T* -> EraseType<T>*
  template <
      typename T, typename T0 = std::remove_pointer_t<T>,
      std::enable_if_t<std::is_pointer<T>::value && is_cast_by_erasure<T0>::value, bool> = true>
  auto _get_ptr() noexcept
      -> decltype(std::declval<Value&>().get_erased_ptr(std::declval<EraseType<T0>*>())) {
    return get_erased_ptr(static_cast<EraseType<T0>*>(nullptr));
  }

  // const T* -> const EraseType<T>*
  template <typename T, typename T0 = std::remove_const_t<std::remove_pointer_t<T>>,
            std::enable_if_t<detail::is_pointer_to_const<T>::value && is_cast_by_erasure<T0>::value,
                             bool> = true>
  auto _get_ptr() const noexcept
      -> decltype(std::declval<Value&>().get_erased_ptr(std::declval<const EraseType<T0>*>())) {
    return get_erased_ptr(static_cast<const EraseType<T0>*>(nullptr));
  }

  template <typename T>
  static auto test_get_ptr(T) -> decltype(std::declval<Value&>()._get_ptr<T>(), std::true_type{});

  static std::false_type test_get_ptr(...);

  template <typename T>
  using has_get_ptr = decltype(test_get_ptr(std::declval<std::add_pointer_t<T>>()));

  template <typename T, std::enable_if_t<std::is_reference<T>::value, bool> = true>
  auto _get_ref() -> decltype((get_ref_impl<T>(std::declval<Value&>()))) {
    return get_ref_impl<T>(*this);
  }

  template <typename T, std::enable_if_t<detail::is_const_reference<T>::value, bool> = true>
  auto _get_ref() const -> decltype((get_ref_impl<T>(std::declval<Value&>()))) {
    return get_ref_impl<T>(*this);
  }

  template <typename T,
            std::enable_if_t<std::is_same<std::remove_const_t<T>, Value>::value, bool> = true>
  Value _get() const {
    return *this;
  }

  template <typename T,
            std::enable_if_t<!std::is_arithmetic<T>::value && has_get_ptr<T>::value, bool> = true>
  auto _get() const
      -> std::remove_reference_t<decltype(std::declval<Value&>()._get_ref<const T&>())> {
    return get_ref<const T&>();
  }

  template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
  T _get() const {
    switch (_type()) {
      case kInt:
        return static_cast<T>(*_get_ptr<const Integer*>());
      case kUInt:
        return static_cast<T>(*_get_ptr<const Unsigned*>());
      case kFloat:
        return static_cast<T>(*_get_ptr<const Float*>());
      case kBool:
        return static_cast<T>(*_get_ptr<const Boolean*>());
      default:
        throw_exception(eInvalidArgument);
    }
  }

  template <typename T, std::enable_if_t<std::is_same<T, const char*>::value, bool> = true>
  const char* _get() const {
    if (_is_string()) {
      return data_.string->c_str();
    }
    throw_exception(eInvalidArgument);
  }

  template <typename T>
  T& _get_to(T& v) const {
    v = get<T>();
    return v;
  }

 public:
  template <typename T>
  auto get_ptr() noexcept -> decltype(std::declval<Value&>()._get_ptr<T>()) {
    return _unwrap()._get_ptr<T>();
  }

  template <typename T>
  auto get_ptr() const noexcept -> decltype(std::declval<const Value&>()._get_ptr<T>()) {
    return _unwrap()._get_ptr<T>();
  }

  template <typename T>
  auto get_ref() -> decltype((std::declval<Value&>()._get_ref<T>())) {
    return _unwrap()._get_ref<T>();
  }

  template <typename T>
  auto get_ref() const -> decltype((std::declval<const Value&>()._get_ref<T>())) {
    return _unwrap()._get_ref<T>();
  }

  template <typename T>
  auto get() -> decltype(std::declval<Value&>()._get<T>()) {
    return _unwrap()._get<T>();
  }

  template <typename T>
  auto get() const -> decltype(std::declval<const Value&>()._get<T>()) {
    return _unwrap()._get<T>();
  }

  template <typename T>
  auto get_to(T& v) const -> decltype((std::declval<const Value&>()._get_to(v))) {
    return _unwrap()._get_to(v);
  }

  Array& array() & { return get_ref<Array&>(); }
  Array&& array() && { return static_cast<Array&&>(get_ref<Array&>()); }
  const Array& array() const& { return get_ref<const Array&>(); }
  const Array&& array() const&& { return static_cast<const Array&&>(get_ref<const Array&>()); }

  Object& object() & { return get_ref<Object&>(); }
  Object&& object() && { return static_cast<Object&&>(get_ref<Object&>()); }
  const Object& object() const& { return get_ref<const Object&>(); }
  const Object&& object() const&& { return static_cast<const Object&&>(get_ref<const Object&>()); }

  value_type& operator[](size_t idx) & {
    return static_cast<value_type&>(_unwrap()._subscript(idx));
  }

  value_type&& operator[](size_t idx) && {
    return static_cast<value_type&&>(_unwrap()._subscript(idx));
  }

  const value_type& operator[](size_t idx) const& {
    return static_cast<const value_type&>(_unwrap()._subscript(idx));
  }

  const value_type&& operator[](size_t idx) const&& {
    return static_cast<const value_type&&>(_unwrap()._subscript(idx));
  }

  value_type& operator[](const Object::key_type& idx) & {
    return static_cast<value_type&>(_unwrap()._subscript(idx));
  }

  value_type&& operator[](const Object::key_type& idx) && {
    return static_cast<value_type&&>(_unwrap()._subscript(idx));
  }

  const value_type& operator[](const Object::key_type& idx) const& {
    return static_cast<const value_type&>(_unwrap()._subscript(idx));
  }

  const value_type&& operator[](const Object::key_type& idx) const&& {
    return static_cast<const value_type&&>(_unwrap()._subscript(idx));
  }

  reference front() { return _unwrap()._front(); }

  const_reference front() const { return _unwrap()._front(); }

  reference back() { return _unwrap()._back(); }

  const_reference back() const { return _unwrap()._back(); }

  void push_back(Value&& val) { _unwrap()._push_back(std::move(val)); }

  void push_back(const Value& val) { _unwrap()._push_back(val); }

  template <typename Key>
  bool contains(Key&& key) const {
    return _unwrap()._contains(std::forward<Key>(key));
  }

  template <typename Key>
  iterator find(Key&& key) {
    return _unwrap()._find(std::forward<Key>(key));
  }

  template <typename Key>
  const_iterator find(Key&& key) const {
    return _unwrap()._find(std::forward<Key>(key));
  }

  template <typename T>
  T value(const typename Object::key_type& key, const T& default_value) const {
    return _unwrap()._value(key, default_value);
  }

  iterator begin() { return _unwrap()._begin(); }

  iterator end() { return _unwrap()._end(); }

  const_iterator begin() const { return _unwrap()._begin(); }

  const_iterator end() const { return _unwrap()._end(); }

  void update(const_reference v) { return _unwrap()._update(v); }

 private:
  reference _front() {
    if (_is_array()) {
      return (*data_.array).front();
    }
    throw_exception(eInvalidArgument);
  }

  const_reference _front() const {
    if (_is_array()) {
      return (*data_.array).front();
    }
    throw_exception(eInvalidArgument);
  }

  reference _back() {
    if (_is_array()) {
      return (*data_.array).back();
    }
    throw_exception(eInvalidArgument);
  }

  const_reference _back() const {
    if (_is_array()) {
      return (*data_.array).back();
    }
    throw_exception(eInvalidArgument);
  }

  void _push_back(Value&& val) {
    if (!(_is_null() || _is_array())) {
      throw_exception(eInvalidArgument);
    }
    if (_is_null()) {
      *this = Type::kArray;
    }
    data_.array->push_back(std::move(val));
  }

  void _push_back(const Value& val) {
    if (!(_is_null() || _is_array())) {
      throw_exception(eInvalidArgument);
    }
    if (_is_null()) {
      *this = Type::kArray;
    }
    data_.array->push_back(val);
  }

  template <typename Key>
  bool _contains(Key&& key) const {
    return _is_object() && data_.object->find(std::forward<Key>(key)) != data_.object->end();
  }

  template <typename Key>
  iterator _find(Key&& key) {
    if (_is_object()) {
      auto iter = data_.object->find(std::forward<Key>(key));
      return {this, iter};
    }
    throw_exception(eInvalidArgument);
  }

  template <typename Key>
  const_iterator _find(Key&& key) const {
    if (_is_object()) {
      auto iter = data_.object->find(std::forward<Key>(key));
      return {this, iter};
    }
    throw_exception(eInvalidArgument);
  }

  template <typename T>
  T _value(const typename Object::key_type& key, const T& default_value) const {
    if (_is_object()) {
      const auto it = _find(key);
      if (it != _end()) {
        return (*it)._get<T>();
      }
      return default_value;
    }
    throw_exception(eInvalidArgument);
  }

  iterator _begin() {
    if (_is_array()) {
      return {this, data_.array->begin()};
    } else if (_is_object()) {
      return {this, data_.object->begin()};
    } else {
      throw_exception(eInvalidArgument);
    }
  }

  iterator _end() {
    if (_is_array()) {
      return {this, data_.array->end()};
    } else if (_is_object()) {
      return {this, data_.object->end()};
    } else {
      throw_exception(eInvalidArgument);
    }
  }

  const_iterator _begin() const {
    if (_is_array()) {
      return {this, data_.array->begin()};
    } else if (_is_object()) {
      return {this, data_.object->begin()};
    } else {
      throw_exception(eInvalidArgument);
    }
  }

  const_iterator _end() const {
    if (_is_array()) {
      return {this, data_.array->end()};
    } else if (_is_object()) {
      return {this, data_.object->end()};
    } else {
      throw_exception(eInvalidArgument);
    }
  }

  void _update(const_reference v) {
    if (_is_null()) {
      type_ = ValueType::kObject;
      data_.object = create<Object>();
    }
    if (!(_is_object() && v._is_object())) {
      throw_exception(eInvalidArgument);
    }
    for (auto it = v._begin(); it != v._end(); ++it) {
      data_.object->operator[](it.key()) = *it;
    }
  }

  Value& _unwrap() {
    auto p = this;
    while (p->_is_pointer() && *p->data_.pointer) {
      p = p->data_.pointer->get();
    }
    return *p;
  }

  const Value& _unwrap() const {
    auto p = this;
    while (p->_is_pointer() && *p->data_.pointer) {
      p = p->data_.pointer->get();
    }
    return *p;
  }

 private:
  template <typename T, typename... Args>
  static T* create(Args&&... args) {
    return new T(std::forward<Args>(args)...);
  }

  template <typename T>
  static void release(T* ptr) {
    delete ptr;
  }

  value_type& _subscript(size_t idx) {
    if (_is_array()) {
      return (*data_.array)[idx];
    }
    throw_exception(eInvalidArgument);
  }

  const value_type& _subscript(size_t idx) const {
    if (_is_array()) {
      return (*data_.array)[idx];
    }
    throw_exception(eInvalidArgument);
  }

  reference _subscript(const Object::key_type& key) {
    if (_is_null()) {
      type_ = Type::kObject;
      data_.object = create<Object>();
    }
    if (_is_object()) {
      return (*data_.object)[key];
    }
    throw_exception(eInvalidArgument);
  }

  const_reference _subscript(const Object::key_type& key) const {
    if (_is_object()) {
      return (*data_.object)[key];
    }
    throw_exception(eInvalidArgument);
  }

 private:
  union ValueData {
    Boolean boolean;
    Integer number_integer;
    Unsigned number_unsigned;
    Float number_float;
    String* string;
    Binary* binary;
    Array* array;
    Object* object;
    Dynamic* dynamic;
    Pointer* pointer;
    Any* any;

    ValueData() = default;

    ValueData(Boolean v) noexcept : boolean(v) {}

    ValueData(Integer v) noexcept : number_integer(v) {}

    ValueData(Unsigned v) noexcept : number_unsigned(v) {}

    ValueData(Float v) noexcept : number_float(v) {}

    ValueData(Type type) {
      switch (type) {
        case Type::kBool:
          boolean = Boolean{};
          break;
        case Type::kInt:
          number_integer = Integer{};
          break;
        case Type::kUInt:
          number_unsigned = Unsigned{};
          break;
        case Type::kFloat:
          number_float = Float{};
          break;
        case Type::kString:
          string = create<String>();
          break;
        case Type::kBinary:
          binary = create<Binary>();
          break;
        case Type::kArray:
          array = create<Array>();
          break;
        case Type::kObject:
          object = create<Object>();
          break;
        case Type::kPointer:
          pointer = create<Pointer>();
          break;
        case Type::kAny:
          any = create<Any>();
          break;
        case Type::kNull:
          object = nullptr;
          break;
        default:
          throw_exception(eNotSupported);
      }
    }

    ValueData(const String& value) { string = create<String>(value); }

    ValueData(String&& value) { string = create<String>(std::move(value)); }

    ValueData(const Binary& value) { binary = create<Binary>(value); }

    ValueData(Binary&& value) { binary = create<Binary>(std::move(value)); }

    ValueData(const Object& value) { object = create<Object>(value); }

    ValueData(Object&& value) { object = create<Object>(std::move(value)); }

    ValueData(const Array& value) { array = create<Array>(value); }

    ValueData(Array&& value) { array = create<Array>(std::move(value)); }

    ValueData(const Pointer& value) { pointer = create<Pointer>(value); }

    ValueData(Pointer&& value) { pointer = create<Pointer>(std::move(value)); }

    // nlohmann/json used an iterative implementation
    void destroy(ValueType t) {
      switch (t) {
        case ValueType::kString:
          release(string);
          break;
        case ValueType::kBinary:
          release(binary);
          break;
        case ValueType::kArray:
          release(array);
          break;
        case ValueType::kObject:
          release(object);
          break;
        case ValueType::kPointer:
          release(pointer);
          break;
        case ValueType::kAny:
          release(any);
          break;
        default:
          break;
      }
    }
  };

  ValueType type_ = ValueType::kNull;
  ValueData data_ = {};
};

namespace detail {

class ValueRef {
 public:
  ValueRef(Value&& value)
      : owned_value_(std::move(value)), value_ref_(&owned_value_), is_rvalue_(true) {}

  ValueRef(const Value& value) : value_ref_(const_cast<Value*>(&value)), is_rvalue_(false) {}

  ValueRef(std::initializer_list<ValueRef> init)
      : owned_value_(init), value_ref_(&owned_value_), is_rvalue_(true) {}

  template <typename... Args, std::enable_if_t<std::is_constructible_v<Value, Args...>, int> = 0>
  ValueRef(Args&&... args)
      : owned_value_(std::forward<Args>(args)...), value_ref_(&owned_value_), is_rvalue_(true) {}

  ValueRef(ValueRef&&) = default;
  ValueRef(const ValueRef&) = delete;
  ValueRef& operator=(const ValueRef&) = delete;
  ValueRef& operator=(ValueRef&&) = delete;
  ~ValueRef() = default;

  Value moved_or_copied() const {
    if (is_rvalue_) {
      return std::move(*value_ref_);
    }
    return *value_ref_;
  }

  const Value& operator*() const { return *static_cast<const Value*>(value_ref_); }
  const Value* operator->() const { return static_cast<const Value*>(value_ref_); }

 private:
  mutable Value owned_value_;
  Value* value_ref_ = nullptr;
  const bool is_rvalue_ = true;
};

}  // namespace detail

inline Value::Value(std::initializer_list<ValueRef> init, bool type_deduction, Type manual_type) {
  bool is_an_object = true;
  for (const auto& x : init) {
    if (!(x->_is_array() && x->_size() == 2 && x->_front()._is_string())) {
      is_an_object = false;
      break;
    }
  }
  if (!type_deduction) {
    if (manual_type == Type::kArray) {
      is_an_object = false;
    }
    if (manual_type == Type::kObject && !is_an_object) {
      throw_exception(eInvalidArgument);
    }
  }
  if (is_an_object) {
    type_ = Type::kObject;
    data_ = Type::kObject;
    for (const auto& x : init) {
      auto e = x.moved_or_copied();
      data_.object->emplace(std::move(*((*e.data_.array)[0].data_.string)),
                            std::move((*e.data_.array)[1]));
    }
  } else {
    type_ = Type::kArray;
    data_.array = create<Array>(init.begin(), init.end());
  }
}

inline Value make_pointer(Value v) { return std::make_shared<Value>(std::move(v)); }

}  // namespace mmdeploy

#endif  // MMDEPLOY_TYPES_VALUE_H_
