// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_DEVICE_CUDA_DEFAULT_ALLOCATOR_H_
#define MMDEPLOY_SRC_DEVICE_CUDA_DEFAULT_ALLOCATOR_H_

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>

#include "mmdeploy/core/logger.h"

namespace mmdeploy::cuda {

class DefaultAllocator {
 public:
  DefaultAllocator() = default;
  ~DefaultAllocator() {
    MMDEPLOY_ERROR("=== CUDA Default Allocator ===");
    MMDEPLOY_ERROR("  Allocation: count={}, size={}MB, time={}ms", alloc_count_,
                   alloc_size_ / (1024 * 1024.f), alloc_time_ / 1000000.f);
    MMDEPLOY_ERROR("Deallocation: count={}, size={}MB, time={}ms", dealloc_count_,
                   dealloc_size_ / (1024 * 1024.f), dealloc_time_ / 1000000.f);
  }
  [[nodiscard]] void* Allocate(std::size_t n) {
    void* p{};
    auto t0 = std::chrono::high_resolution_clock::now();
    auto ret = cudaMalloc(&p, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    alloc_time_ += (int64_t)std::chrono::duration<double, std::nano>(t1 - t0).count();
    if (ret != cudaSuccess) {
      MMDEPLOY_ERROR("error allocating cuda memory: {}", cudaGetErrorString(ret));
      return nullptr;
    }
    alloc_count_ += 1;
    alloc_size_ += n;
    return p;
  }
  void Deallocate(void* p, std::size_t n) {
    (void)n;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto ret = cudaFree(p);
    auto t1 = std::chrono::high_resolution_clock::now();
    dealloc_time_ += (int64_t)std::chrono::duration<double, std::nano>(t1 - t0).count();
    if (ret != cudaSuccess) {
      MMDEPLOY_ERROR("error deallocating cuda memory: {}", cudaGetErrorString(ret));
      return;
    }
    dealloc_count_ += 1;
    dealloc_size_ += n;
  }

 private:
  std::atomic<std::size_t> alloc_count_;
  std::atomic<std::size_t> alloc_size_;
  std::atomic<std::size_t> alloc_time_;
  std::atomic<std::size_t> dealloc_count_;
  std::atomic<std::size_t> dealloc_size_;
  std::atomic<std::size_t> dealloc_time_;
};

inline DefaultAllocator& gDefaultAllocator() {
  static DefaultAllocator v;
  return v;
}

}  // namespace mmdeploy::cuda

#endif  // MMDEPLOY_SRC_DEVICE_CUDA_DEFAULT_ALLOCATOR_H_
