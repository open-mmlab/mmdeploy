// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_DEVICE_CUDA_BUDDY_ALLOCATOR_H_
#define MMDEPLOY_SRC_DEVICE_CUDA_BUDDY_ALLOCATOR_H_

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <list>
#include <mutex>
#include <vector>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/device/cuda/default_allocator.h"

namespace mmdeploy::cuda {

class BuddyAllocator {
 public:
  using size_type = std::size_t;

  BuddyAllocator(size_type size, size_type block_size) {
    block_size_ = block_size;
    block_count_ = size / block_size_;
    if (!IsPowerOfTwo(block_count_)) {
      block_count_ = RoundToPowerOfTwo(block_count_);
      MMDEPLOY_WARN("Rounding up block_count to next power of 2 {}", block_count_);
    }
    base_ = LogPowerOfTwo(block_count_);
    size_ = block_size_ * block_count_;
    memory_ = gDefaultAllocator().Allocate(size_);
    tree_.resize(block_count_ * 2);
    free_.resize(base_ + 1);
    Build(1, 0);
    Add(1, 0);
    MMDEPLOY_ERROR("size = {}, block_size = {}, block_count = {}", size_, block_size_,
                   block_count_);
    size = size_;
    for (int i = 0; i <= base_; ++i) {
      MMDEPLOY_ERROR("level {}, size = {}", i, size);
      size /= 2;
    }
  }

  ~BuddyAllocator() {
    for (int i = 0; i < free_.size(); ++i) {
      MMDEPLOY_ERROR("free_[{}].size(): {}", i, free_[i].size());
    }
    gDefaultAllocator().Deallocate(memory_, size_);
  }

  [[nodiscard]] void* Allocate(size_type n) {
    std::lock_guard lock{mutex_};
    if (n > size_) {
      return nullptr;
    }
    auto n_level = GetLevel(n);
    auto level = n_level;
    for (; level >= 0; --level) {
      if (!free_[level].empty()) {
        break;
      }
    }
    if (level < 0) {
      MMDEPLOY_WARN("failed to allocate memory size = {} bytes", n);
      return nullptr;
    }
    for (; level < n_level; ++level) {
      auto index = free_[level].front();
      Split(index, level);
    }
    auto index = free_[level].front();
    Del(index, level);
    auto offset = (index ^ (1 << level)) << (base_ - level);
    auto p = static_cast<uint8_t*>(memory_) + offset * block_size_;
    return p;
  }

  void Deallocate(void* p, size_type n) {
    std::lock_guard lock{mutex_};
    auto offset = static_cast<uint8_t*>(p) - static_cast<uint8_t*>(memory_);
    if (offset < 0 || offset % block_size_) {
      MMDEPLOY_ERROR("invalid address: {}", p);
    }
    offset /= static_cast<long>(block_size_);
    auto level = GetLevel(n);
    auto index = (offset >> (base_ - level)) ^ (1 << level);
    Add(index, level);
    while (index > 1) {
      auto buddy = index ^ 1;
      if (tree_[buddy] != free_[level].end()) {
        Merge(index, level);
        index /= 2;
        --level;
      } else {
        break;
      }
    }
  }

 private:
  void Add(size_type index, size_type level) {
    assert(tree_[index] == free_[level].end());
    tree_[index] = free_[level].insert(free_[level].end(), index);
  }

  void Del(size_type index, size_type level) {
    assert(tree_[index] != free_[level].end());
    free_[level].erase(tree_[index]);
    tree_[index] = free_[level].end();
  }

  void Split(size_type index, size_type level) {
    Del(index, level);
    Add(index * 2, level + 1);
    Add(index * 2 + 1, level + 1);
  }

  void Merge(size_type index, size_type level) {
    Del(index, level);
    Del(index ^ 1, level);
    Add(index / 2, level - 1);
  }

  size_type GetLevel(size_type size) const {
    size = RoundToPowerOfTwo((size + block_size_ - 1) / block_size_);
    return base_ - LogPowerOfTwo(size);
  }

  static bool IsPowerOfTwo(size_type n) { return (n & (n - 1)) == 0; }

  static size_type RoundToPowerOfTwo(size_type n) {
    --n;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    n |= (n >> 32);
    return ++n;
  }

  static size_type LogPowerOfTwo(size_type v) {
    size_type r{};
    r |= ((v & 0xFFFFFFFF00000000) != 0) << 5;
    r |= ((v & 0xFFFF0000FFFF0000) != 0) << 4;
    r |= ((v & 0xFF00FF00FF00FF00) != 0) << 3;
    r |= ((v & 0xF0F0F0F0F0F0F0F0) != 0) << 2;
    r |= ((v & 0xCCCCCCCCCCCCCCCC) != 0) << 1;
    r |= ((v & 0xAAAAAAAAAAAAAAAA) != 0);
    return r;
  }

  void Build(size_type index, size_type level) {
    if (index < tree_.size()) {
      tree_[index] = free_[level].end();
      index *= 2;
      ++level;
      Build(index, level);
      Build(index + 1, level);
    }
  }

 private:
  size_type size_;
  size_type block_size_;
  size_type block_count_;
  size_type base_;
  void* memory_;
  std::vector<std::list<size_type>::iterator> tree_;
  std::vector<std::list<size_type> > free_;
  std::mutex mutex_;
};

inline BuddyAllocator& gBuddyAllocator() {
  static BuddyAllocator v(1U << 30, 1024 * 64);
  return v;
}

}  // namespace mmdeploy::cuda

#endif  // MMDEPLOY_SRC_DEVICE_CUDA_BUDDY_ALLOCATOR_H_
