// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MPL_SPAN_H_
#define MMDEPLOY_SRC_CORE_MPL_SPAN_H_

#include <iterator>
#include <type_traits>

#include "detected.h"
#include "iterator.h"

namespace mmdeploy {

namespace detail {

template <typename T>
using arrow_t = decltype(std::declval<T>().operator->());

template <typename T>
constexpr auto to_address(const T& p) noexcept {
  if constexpr (std::is_pointer_v<T>) {
    return p;
  } else if (detail::is_detected_v<arrow_t, T>) {
    return to_address(p.operator->());
  }
}

}  // namespace detail

template <typename T>
class Span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using reverse_iterator = std::reverse_iterator<iterator>;

 public:
  constexpr Span() noexcept : data_(nullptr), size_(0) {}

  // clang-format off
  template <typename It,
      std::void_t<decltype(std::addressof(std::declval<It&>()))>* = nullptr>
  // clang-format on
  constexpr Span(It first, size_type size) : data_(detail::to_address(first)), size_(size) {}

  template <typename It, typename End,
            std::enable_if_t<!std::is_convertible_v<End, std::size_t>, int> = 0>
  constexpr Span(It first, End last) : data_(detail::to_address(first)), size_(last - first) {}

  template <typename U, typename = std::void_t<decltype(std::data(std::declval<U>()))>,
            typename = std::void_t<decltype(std::size(std::declval<U>()))>>
  constexpr Span(U& v) : data_(std::data(v)), size_(std::size(v)) {}

  template <typename U, typename = std::void_t<decltype(std::data(std::declval<U>()))>,
            typename = std::void_t<decltype(std::size(std::declval<U>()))>>
  constexpr Span(const U& v) : data_(std::data(v)), size_(std::size(v)) {}

  template <typename U>
  constexpr Span(std::initializer_list<U> il) noexcept : Span(il.begin(), il.size()) {}

  template <std::size_t N>
  constexpr Span(element_type (&arr)[N]) noexcept : data_(std::data(arr)), size_(N) {}

  constexpr Span(const Span& other) noexcept : data_(std::data(other)), size_(std::size(other)) {}

  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }
  constexpr reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
  constexpr reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }
  constexpr reference front() const { return data_[0]; }
  constexpr reference back() const { return data_[size_ - 1]; }
  constexpr reference operator[](size_type idx) const { return data_[idx]; }
  constexpr pointer data() const noexcept { return data_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type size_bytes() const noexcept { return sizeof(value_type) * size(); }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr Span<element_type> first(size_type count) const { return {begin(), count}; }
  constexpr Span<element_type> last(size_type count) const { return {end() - count, count}; }
  constexpr Span<element_type> subspan(size_type offset, size_type count = -1) const {
    if (count == -1) {
      return Span(begin() + offset, end());
    } else {
      return Span(begin() + offset, begin() + offset + count);
    }
  }

  constexpr Span& operator=(const Span& other) noexcept = default;

  template <typename U>
  friend bool operator!=(const Span& a, const Span<U>& b) {
    if (a.size() != b.size()) {
      return true;
    }
    for (size_type i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) {
        return true;
      }
    }
    return false;
  }

  template <typename U>
  friend bool operator==(const Span& a, const Span<U>& b) {
    return !(a != b);
  }

 private:
  T* data_;
  size_type size_;
};
// clang-format off
template <typename It, typename EndOrSize>
Span(It, EndOrSize) -> Span<std::remove_reference_t<iter_reference_t<It>>>;

template <typename T, std::size_t N>
Span(T (&)[N]) -> Span<T>;

template <typename U, typename = std::void_t<decltype(std::declval<U>().data())>,
          typename = std::void_t<decltype(std::declval<U>().size())>>
Span(U& v) -> Span<typename uncvref_t<U>::value_type>;

template <typename T>
Span(std::initializer_list<T>) -> Span<const T>;
// clang-format on
}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_MPL_SPAN_H_
