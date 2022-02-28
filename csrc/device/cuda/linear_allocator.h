// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_DEVICE_CUDA_LINEARALLOCATOR_H_
#define MMDEPLOY_SRC_DEVICE_CUDA_LINEARALLOCATOR_H_

#include "default_allocator.h"

namespace mmdeploy::cuda {

class LinearAllocator {
 public:
  explicit LinearAllocator(std::size_t size) : size_(size) {
    base_ = static_cast<uint8_t*>(gDefaultAllocator().Allocate(size));
    ptr_ = base_;
  }
  ~LinearAllocator() { gDefaultAllocator().Deallocate(base_, size_); }
  [[nodiscard]] void* Allocate(std::size_t n) {
    std::optional<std::lock_guard<std::mutex> > lock;
    if (mutex_) {
      lock.emplace(*mutex_);
    }
    ++count_;
    total_ += n;
    auto ptr = static_cast<void*>(ptr_);
    std::size_t space = base_ + size_ - ptr_;

    if (std::align(16, n, ptr, space)) {
      MMDEPLOY_ERROR("success n={}, total={}, count={}", n, total_, count_);
      ptr_ = static_cast<uint8_t*>(ptr) + n;
      return ptr;
    }
    MMDEPLOY_ERROR("fallback {}, total={}, count={}", n, total_, count_);
    return gDefaultAllocator().Allocate(n);
  }
  void Deallocate(void* _p, std::size_t n) {
    std::optional<std::lock_guard<std::mutex> > lock;
    if (mutex_) {
      lock.emplace(*mutex_);
    }
    auto p = static_cast<uint8_t*>(_p);
    if (!(base_ <= p && p < ptr_)) {
      gDefaultAllocator().Deallocate(_p, n);
    }
    total_ -= n;
    --count_;
    MMDEPLOY_ERROR("deallocate total={}, count={}", total_, count_);
    if (total_ == 0) {
      assert(count_ == 0);
      ptr_ = base_;
    }
  }

 private:
  std::size_t size_;
  uint8_t* base_;
  uint8_t* ptr_;
  std::size_t total_{};
  std::size_t count_{};
  std::optional<std::mutex> mutex_;
};

inline LinearAllocator& gLinearAllocator() {
  static LinearAllocator v(1U << 30);
  return v;
}

}  // namespace mmdeploy::cuda

#endif  // MMDEPLOY_SRC_DEVICE_CUDA_LINEARALLOCATOR_H_
