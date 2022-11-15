// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_CORE_MPL_STRUCTURE_H_
#define MMDEPLOY_CSRC_MMDEPLOY_CORE_MPL_STRUCTURE_H_

#include <array>
#include <memory>
#include <tuple>
#include <utility>

namespace mmdeploy {

namespace _structure {

using std::array;
using std::index_sequence;
using std::integral_constant;
using std::tuple;

// [p0][T0]...[p1][T1]...[pn][Tn]...[px][X]
// ^                                     |
// |-------------------------------------|
template <size_t Size>
class Storage {
  static constexpr auto S = Size + 1;
  using Indices = std::make_index_sequence<S>;

 public:
  Storage(const Storage&) = delete;
  Storage(Storage&&) noexcept = delete;
  Storage& operator=(const Storage&) = delete;
  Storage& operator=(Storage&&) noexcept = delete;

  Storage(const array<size_t, Size>& sizes, const array<size_t, Size>& aligns) {
    create(std::make_index_sequence<Size>{}, sizes, aligns);
  }

  template <size_t offset>
  Storage(const array<size_t, Size>& sizes, const array<size_t, Size>& aligns,
          integral_constant<size_t, offset> index, void* ptr) noexcept {
    create(std::make_index_sequence<Size>{}, sizes, aligns, index, ptr);
  }

  template <size_t... i, typename... As>
  void create(index_sequence<i...>, const array<size_t, Size>& sizes,
              const array<size_t, Size>& aligns, As&&... as) {
    std::tie(data_, pointers_) =
        Creator{{sizes[i]..., sizeof(void*)}, {aligns[i]..., alignof(void*)}}.create((As &&) as...);
  }

  ~Storage() {
    if (data_) {
      delete[] static_cast<uint8_t*>(data_);
      release();
    }
  }

  void* data() const noexcept { return data_; }

  template <size_t i>
  void* at() const noexcept {
    return pointers_[i];
  }

  array<void*, S>& pointers() { return pointers_; }

  void* release() noexcept {
    std::fill_n(pointers_.data(), S, nullptr);
    return std::exchange(data_, nullptr);
  }

 private:
  struct Creator {
    const array<size_t, S>& sizes_;
    const array<size_t, S>& aligns_;

    tuple<void*, array<void*, S>> create() {
      auto space = get_space(Indices{});
      void* data = new uint8_t[space];
      auto ptr = data;
      array<void*, S> pointers{};
      // build the layout according to sizes and alignments
      align<0>(ptr, space, pointers, Indices{});
      // store a pointer to the head of data in the last slot
      *reinterpret_cast<void**>(pointers.back()) = data;
      return {data, pointers};
    }

    template <size_t offset>
    tuple<void*, array<void*, S>> create(integral_constant<size_t, offset>, void* ptr) {
      auto space = get_space(Indices{});
      array<void*, S> pointers{};
      // recover the layout after offset
      align<offset>(ptr, space, pointers, std::make_index_sequence<S - offset>{});
      // recover data pointer
      auto data = ptr = *reinterpret_cast<void**>(pointers.back());
      // recover the layout before offset
      align<0>(ptr, space, pointers, std::make_index_sequence<offset>{});
      return {data, pointers};
    }

   private:
    template <size_t... i>
    size_t get_space(index_sequence<i...>) const noexcept {
      return ((sizes_[i] + aligns_[i]) + ...);
    }

    template <size_t offset, size_t... i>
    void align(void*& ptr, size_t& space, array<void*, S>& pointers,
               index_sequence<i...>) noexcept {
      (align(ptr, space, pointers, integral_constant<size_t, offset + i>{}), ...);
    }

    template <size_t i>
    void align(void*& ptr, size_t& space, array<void*, S>& pointers,
               integral_constant<size_t, i>) noexcept {
      pointers[i] = std::align(aligns_[i], sizes_[i], ptr, space);
      ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + sizes_[i]);
      space -= sizes_[i];
    }
  };

 private:
  void* data_{};
  array<void*, S> pointers_{};
};

template <typename T, typename... Ts>
struct _count {
  static constexpr size_t value = (std::is_same_v<T, Ts> + ...);
};

template <typename T, typename Ts, typename Is, typename = void>
struct get_type_index {};

template <typename T, typename... Ts, size_t... Is>
struct get_type_index<T, tuple<Ts...>, std::index_sequence<Is...>,
                      std::enable_if_t<_count<T, Ts...>::value == 1>> {
  static constexpr size_t value = ((std::is_same_v<T, Ts> * Is) + ...);
};

template <typename T>
using _size_t = size_t;

template <typename... Ts>
class Structure : public Storage<sizeof...(Ts)> {
  static constexpr auto Size = sizeof...(Ts);
  using Base = Storage<Size>;
  using Indices = std::index_sequence_for<Ts...>;

 public:
  explicit Structure() : Structure(1) {}

  explicit Structure(size_t length) : Structure(array<size_t, Size>{_size_t<Ts>(length)...}) {}

  explicit Structure(const array<size_t, Size>& lengths)
      : Base(get_sizes(lengths, Indices{}), {alignof(Ts)...}), lengths_{lengths} {
    construct(Indices{});
  }

  template <typename T, size_t index = get_type_index<T, tuple<Ts...>, Indices>::value>
  explicit Structure(T* p) : Structure(1, integral_constant<size_t, index>{}, p) {}

  template <typename T, size_t index = get_type_index<T, tuple<Ts...>, Indices>::value>
  explicit Structure(size_t length, T* p)
      : Structure(length, integral_constant<size_t, index>{}, p) {}

  template <typename T, size_t index = get_type_index<T, tuple<Ts...>, Indices>::value>
  explicit Structure(const array<size_t, Size>& lengths, T* p)
      : Structure(lengths, integral_constant<size_t, index>{}, p) {}

  template <size_t i>
  explicit Structure(integral_constant<size_t, i> index, void* p) : Structure(1, index, p) {}

  template <size_t i>
  explicit Structure(size_t length, integral_constant<size_t, i> index, void* p)
      : Structure({_size_t<Ts>(length)...}, index, p) {}

  template <size_t i>
  explicit Structure(const array<size_t, Size>& lengths, integral_constant<size_t, i> index,
                     void* p)
      : Base(get_sizes(lengths, Indices{}), {alignof(Ts)...}, index, p), lengths_{lengths} {}

  ~Structure() {
    if (this->data()) {
      destruct(Indices{});
    }
  }

  template <size_t i>
  decltype(auto) get() const {
    using T = std::tuple_element_t<i, tuple<Ts...>>;
    return reinterpret_cast<T*>(this->template at<i>());
  }

  tuple<Ts*...> pointers() const noexcept { return pointers(Indices{}); }

 private:
  template <size_t... i>
  static array<size_t, Size> get_sizes(const array<size_t, Size>& lengths,
                                       index_sequence<i...>) noexcept {
    return {(sizeof(Ts) * lengths[i])...};
  }

  template <size_t... i>
  tuple<Ts*...> pointers(index_sequence<i...>) const noexcept {
    return {get<i>()...};
  }

  template <size_t... i>
  void construct(index_sequence<i...>) {
    (create_n(get<i>(), lengths_[i]), ...);
  }

  template <typename T>
  static void create_n(T* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      new (data + i) T{};
    }
  }

  template <size_t... i>
  void destruct(index_sequence<i...>) {
    (std::destroy_n(get<i>(), lengths_[i]), ...);
  }

 private:
  array<size_t, Size> lengths_;
};

}  // namespace _structure

using _structure::Structure;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_CORE_MPL_STRUCTURE_H_
