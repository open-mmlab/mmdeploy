// Copyright (c) OpenMMLab. All rights reserved.
// Re-implementation of std::any, relies on static type id instead of RTTI.
// adjusted from libc++-10

#ifndef MMDEPLOY_CSRC_CORE_MPL_STATIC_ANY_H_
#define MMDEPLOY_CSRC_CORE_MPL_STATIC_ANY_H_

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "mmdeploy/core/mpl/type_traits.h"

namespace mmdeploy {

namespace detail {

template <typename T>
struct is_in_place_type_impl : std::false_type {};

template <typename T>
struct is_in_place_type_impl<std::in_place_type_t<T>> : std::true_type {};

template <typename T>
struct is_in_place_type : public is_in_place_type_impl<T> {};

}  // namespace detail

class BadAnyCast : public std::bad_cast {
 public:
  const char* what() const noexcept override { return "BadAnyCast"; }
};

[[noreturn]] inline void ThrowBadAnyCast() {
#if __cpp_exceptions
  throw BadAnyCast{};
#else
  std::abort();
#endif
}

// Forward declarations
class StaticAny;

template <class ValueType>
std::add_pointer_t<std::add_const_t<ValueType>> static_any_cast(const StaticAny*) noexcept;

template <class ValueType>
std::add_pointer_t<ValueType> static_any_cast(StaticAny*) noexcept;

namespace __static_any_impl {

using _Buffer = std::aligned_storage_t<3 * sizeof(void*), std::alignment_of_v<void*>>;

template <class T>
using _IsSmallObject =
    std::integral_constant<bool, sizeof(T) <= sizeof(_Buffer) &&
                                     std::alignment_of_v<_Buffer> % std::alignment_of_v<T> == 0 &&
                                     std::is_nothrow_move_constructible_v<T>>;

enum class _Action { _Destroy, _Copy, _Move, _Get, _TypeInfo };

union _Ret {
  void* ptr_;
  traits::type_id_t type_id_;
};

template <class T>
struct _SmallHandler;
template <class T>
struct _LargeHandler;

template <class T>
inline bool __compare_typeid(traits::type_id_t __id) {
  if (__id && __id == traits::TypeId<T>::value) {
    return true;
  }
  return false;
}

template <class T>
using _Handler = std::conditional_t<_IsSmallObject<T>::value, _SmallHandler<T>, _LargeHandler<T>>;

}  // namespace __static_any_impl

class StaticAny {
 public:
  constexpr StaticAny() noexcept : h_(nullptr) {}

  StaticAny(const StaticAny& other) : h_(nullptr) {
    if (other.h_) {
      other.__call(_Action::_Copy, this);
    }
  }

  StaticAny(StaticAny&& other) noexcept : h_(nullptr) {
    if (other.h_) {
      other.__call(_Action::_Move, this);
    }
  }

  template <class ValueType, class T = std::decay_t<ValueType>,
            class = std::enable_if_t<
                !std::is_same<T, StaticAny>::value && !detail::is_in_place_type<ValueType>::value &&
                std::is_copy_constructible<T>::value && traits::TypeId<T>::value>>
  explicit StaticAny(ValueType&& value);

  template <
      class ValueType, class... Args, class T = std::decay_t<ValueType>,
      class = std::enable_if_t<std::is_constructible<T, Args...>::value &&
                               std::is_copy_constructible<T>::value && traits::TypeId<T>::value>>
  explicit StaticAny(std::in_place_type_t<ValueType>, Args&&... args);

  template <class ValueType, class U, class... Args, class T = std::decay_t<ValueType>,
            class = std::enable_if_t<
                std::is_constructible<T, std::initializer_list<U>&, Args...>::value &&
                std::is_copy_constructible<T>::value && traits::TypeId<T>::value>>
  explicit StaticAny(std::in_place_type_t<ValueType>, std::initializer_list<U>, Args&&... args);

  ~StaticAny() { this->reset(); }

  StaticAny& operator=(const StaticAny& rhs) {
    StaticAny(rhs).swap(*this);
    return *this;
  }

  StaticAny& operator=(StaticAny&& rhs) noexcept {
    StaticAny(std::move(rhs)).swap(*this);
    return *this;
  }

  template <
      class ValueType, class T = std::decay_t<ValueType>,
      class = std::enable_if_t<!std::is_same<T, StaticAny>::value &&
                               std::is_copy_constructible<T>::value && traits::TypeId<T>::value>>
  StaticAny& operator=(ValueType&& v);

  template <
      class ValueType, class... Args, class T = std::decay_t<ValueType>,
      class = std::enable_if_t<std::is_constructible<T, Args...>::value &&
                               std::is_copy_constructible<T>::value && traits::TypeId<T>::value>>
  T& emplace(Args&&... args);

  template <class ValueType, class U, class... Args, class T = std::decay_t<ValueType>,
            class = std::enable_if_t<
                std::is_constructible<T, std::initializer_list<U>&, Args...>::value &&
                std::is_copy_constructible<T>::value && traits::TypeId<T>::value>>
  T& emplace(std::initializer_list<U>, Args&&...);

  void reset() noexcept {
    if (h_) {
      this->__call(_Action::_Destroy);
    }
  }

  void swap(StaticAny& rhs) noexcept;

  bool has_value() const noexcept { return h_ != nullptr; }

  traits::type_id_t type() const noexcept {
    if (h_) {
      return this->__call(_Action::_TypeInfo).type_id_;
    } else {
      return traits::TypeId<void>::value;
    }
  }

 private:
  using _Action = __static_any_impl::_Action;
  using _Ret = __static_any_impl::_Ret;
  using _HandleFuncPtr = _Ret (*)(_Action, const StaticAny*, StaticAny*, traits::type_id_t info);

  union _Storage {
    constexpr _Storage() : ptr_(nullptr) {}
    void* ptr_;
    __static_any_impl::_Buffer buf_;
  };

  _Ret __call(_Action a, StaticAny* other = nullptr, traits::type_id_t info = 0) const {
    return h_(a, this, other, info);
  }

  _Ret __call(_Action a, StaticAny* other = nullptr, traits::type_id_t info = 0) {
    return h_(a, this, other, info);
  }

  template <class>
  friend struct __static_any_impl::_SmallHandler;

  template <class>
  friend struct __static_any_impl::_LargeHandler;

  template <class ValueType>
  friend std::add_pointer_t<std::add_const_t<ValueType>> static_any_cast(const StaticAny*) noexcept;

  template <class ValueType>
  friend std::add_pointer_t<ValueType> static_any_cast(StaticAny*) noexcept;

  _HandleFuncPtr h_ = nullptr;
  _Storage s_;
};

namespace __static_any_impl {

template <class T>
struct _SmallHandler {
  static _Ret __handle(_Action action, const StaticAny* self, StaticAny* other,
                       traits::type_id_t info) {
    _Ret ret;
    ret.ptr_ = nullptr;
    switch (action) {
      case _Action::_Destroy:
        __destroy(const_cast<StaticAny&>(*self));
        break;
      case _Action::_Copy:
        __copy(*self, *other);
        break;
      case _Action::_Move:
        __move(const_cast<StaticAny&>(*self), *other);
        break;
      case _Action::_Get:
        ret.ptr_ = __get(const_cast<StaticAny&>(*self), info);
        break;
      case _Action::_TypeInfo:
        ret.type_id_ = __type_info();
        break;
    }
    return ret;
  }

  template <class... Args>
  static T& __create(StaticAny& dest, Args&&... args) {
    T* ret = ::new (static_cast<void*>(&dest.s_.buf_)) T(std::forward<Args>(args)...);
    dest.h_ = &_SmallHandler::__handle;
    return *ret;
  }

 private:
  template <class... Args>
  static void __destroy(StaticAny& self) {
    T& value = *static_cast<T*>(static_cast<void*>(&self.s_.buf_));
    value.~T();
    self.h_ = nullptr;
  }

  template <class... Args>
  static void __copy(const StaticAny& self, StaticAny& dest) {
    _SmallHandler::__create(dest, *static_cast<const T*>(static_cast<const void*>(&self.s_.buf_)));
  }

  static void __move(StaticAny& self, StaticAny& dest) {
    _SmallHandler::__create(dest, std::move(*static_cast<T*>(static_cast<void*>(&self.s_.buf_))));
    __destroy(self);
  }

  static void* __get(StaticAny& self, traits::type_id_t info) {
    if (__static_any_impl::__compare_typeid<T>(info)) {
      return static_cast<void*>(&self.s_.buf_);
    }
    return nullptr;
  }

  static traits::type_id_t __type_info() { return traits::TypeId<T>::value; }
};

template <class T>
struct _LargeHandler {
  static _Ret __handle(_Action action, const StaticAny* self, StaticAny* other,
                       traits::type_id_t info) {
    _Ret ret;
    ret.ptr_ = nullptr;
    switch (action) {
      case _Action::_Destroy:
        __destroy(const_cast<StaticAny&>(*self));
        break;
      case _Action::_Copy:
        __copy(*self, *other);
        break;
      case _Action::_Move:
        __move(const_cast<StaticAny&>(*self), *other);
        break;
      case _Action::_Get:
        ret.ptr_ = __get(const_cast<StaticAny&>(*self), info);
        break;
      case _Action::_TypeInfo:
        ret.type_id_ = __type_info();
        break;
    }
    return ret;
  }

  template <class... Args>
  static T& __create(StaticAny& dest, Args&&... args) {
    using _Alloc = std::allocator<T>;
    _Alloc alloc;
    auto dealloc = [&](T* p) { alloc.deallocate(p, 1); };
    std::unique_ptr<T, decltype(dealloc)> hold(alloc.allocate(1), dealloc);
    T* ret = ::new ((void*)hold.get()) T(std::forward<Args>(args)...);
    dest.s_.ptr_ = hold.release();
    dest.h_ = &_LargeHandler::__handle;
    return *ret;
  }

 private:
  static void __destroy(StaticAny& self) {
    delete static_cast<T*>(self.s_.ptr_);
    self.h_ = nullptr;
  }

  static void __copy(const StaticAny& self, StaticAny& dest) {
    _LargeHandler::__create(dest, *static_cast<const T*>(self.s_.ptr_));
  }

  static void __move(StaticAny& self, StaticAny& dest) {
    dest.s_.ptr_ = self.s_.ptr_;
    dest.h_ = &_LargeHandler::__handle;
    self.h_ = nullptr;
  }

  static void* __get(StaticAny& self, traits::type_id_t info) {
    if (__static_any_impl::__compare_typeid<T>(info)) {
      return static_cast<void*>(self.s_.ptr_);
    }
    return nullptr;
  }

  static traits::type_id_t __type_info() { return traits::TypeId<T>::value; }
};

}  // namespace __static_any_impl

template <class ValueType, class T, class>
StaticAny::StaticAny(ValueType&& v) : h_(nullptr) {
  __static_any_impl::_Handler<T>::__create(*this, std::forward<ValueType>(v));
}

template <class ValueType, class... Args, class T, class>
StaticAny::StaticAny(std::in_place_type_t<ValueType>, Args&&... args) {
  __static_any_impl::_Handler<T>::__create(*this, std::forward<Args>(args)...);
}

template <class ValueType, class U, class... Args, class T, class>
StaticAny::StaticAny(std::in_place_type_t<ValueType>, std::initializer_list<U> il, Args&&... args) {
  __static_any_impl::_Handler<T>::__create(*this, il, std::forward<Args>(args)...);
}

template <class ValueType, class, class>
inline StaticAny& StaticAny::operator=(ValueType&& v) {
  StaticAny(std::forward<ValueType>(v)).swap(*this);
  return *this;
}

template <class ValueType, class... Args, class T, class>
inline T& StaticAny::emplace(Args&&... args) {
  reset();
  return __static_any_impl::_Handler<T>::__create(*this, std::forward<Args>(args)...);
}

template <class ValueType, class U, class... Args, class T, class>
inline T& StaticAny::emplace(std::initializer_list<U> il, Args&&... args) {
  reset();
  return __static_any_impl::_Handler<T>::_create(*this, il, std::forward<Args>(args)...);
}

inline void StaticAny::swap(StaticAny& rhs) noexcept {
  if (this == &rhs) {
    return;
  }
  if (h_ && rhs.h_) {
    StaticAny tmp;
    rhs.__call(_Action::_Move, &tmp);
    this->__call(_Action::_Move, &rhs);
    tmp.__call(_Action::_Move, this);
  } else if (h_) {
    this->__call(_Action::_Move, &rhs);
  } else if (rhs.h_) {
    rhs.__call(_Action::_Move, this);
  }
}

inline void swap(StaticAny& lhs, StaticAny& rhs) noexcept { lhs.swap(rhs); }

template <class T, class... Args>
inline StaticAny make_static_any(Args&&... args) {
  return StaticAny(std::in_place_type<T>, std::forward<Args>(args)...);
}

template <class T, class U, class... Args>
StaticAny make_static_any(std::initializer_list<U> il, Args&&... args) {
  return StaticAny(std::in_place_type<T>, il, std::forward<Args>(args)...);
}

template <class ValueType>
ValueType static_any_cast(const StaticAny& v) {
  using _RawValueType = std::remove_cv_t<std::remove_reference_t<ValueType>>;
  static_assert(std::is_constructible<ValueType, const _RawValueType&>::value,
                "ValueType is required to be a const lvalue reference "
                "or a CopyConstructible type");
  auto tmp = static_any_cast<std::add_const_t<_RawValueType>>(&v);
  if (tmp == nullptr) {
    ThrowBadAnyCast();
  }
  return static_cast<ValueType>(*tmp);
}

template <class ValueType>
inline ValueType static_any_cast(StaticAny& v) {
  using _RawValueType = std::remove_cv_t<std::remove_reference_t<ValueType>>;
  static_assert(std::is_constructible<ValueType, _RawValueType&>::value,
                "ValueType is required to be an lvalue reference "
                "or a CopyConstructible type");
  auto tmp = static_any_cast<_RawValueType>(&v);
  if (tmp == nullptr) {
    ThrowBadAnyCast();
  }
  return static_cast<ValueType>(*tmp);
}

template <class ValueType>
inline ValueType static_any_cast(StaticAny&& v) {
  using _RawValueType = std::remove_cv_t<std::remove_reference_t<ValueType>>;
  static_assert(std::is_constructible<ValueType, _RawValueType>::value,
                "ValueType is required to be an rvalue reference "
                "or a CopyConstructible type");
  auto tmp = static_any_cast<_RawValueType>(&v);
  if (tmp == nullptr) {
    ThrowBadAnyCast();
  }
  return static_cast<ValueType>(std::move(*tmp));
}

template <class ValueType>
inline std::add_pointer_t<std::add_const_t<ValueType>> static_any_cast(
    const StaticAny* __any) noexcept {
  static_assert(!std::is_reference<ValueType>::value, "ValueType may not be a reference.");
  return static_any_cast<ValueType>(const_cast<StaticAny*>(__any));
}

template <class RetType>
inline RetType __pointer_or_func_test(void* p, std::false_type) noexcept {
  return static_cast<RetType>(p);
}

template <class RetType>
inline RetType __pointer_or_func_test(void*, std::true_type) noexcept {
  return nullptr;
}

template <class ValueType>
std::add_pointer_t<ValueType> static_any_cast(StaticAny* any) noexcept {
  using __static_any_impl::_Action;
  static_assert(!std::is_reference<ValueType>::value, "ValueType may not be a reference.");
  using ReturnType = std::add_pointer_t<ValueType>;
  if (any && any->h_) {
    void* p = any->__call(_Action::_Get, nullptr, traits::TypeId<ValueType>::value).ptr_;
    return __pointer_or_func_test<ReturnType>(p, std::is_function<ValueType>{});
  }
  return nullptr;
}

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_CORE_MPL_STATIC_ANY_H_
