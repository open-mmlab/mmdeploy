// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_DEVICE_ALLOCATOR_H_
#define MMDEPLOY_SRC_CORE_DEVICE_ALLOCATOR_H_

#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <stack>

#include "mmdeploy/core/device_impl.h"
#include "mmdeploy/core/logger.h"

namespace mmdeploy::framework::device_allocator {

class Fallback : public AllocatorImpl {
 public:
  Fallback(AllocatorImplPtr primary, AllocatorImplPtr fallback)
      : primary_(std::move(primary)), fallback_(std::move(fallback)) {}

  Block Allocate(size_t size) noexcept override {
    if (auto block = primary_->Allocate(size); block.handle) {
      return block;
    }
    return fallback_->Allocate(size);
  }

  void Deallocate(Block& block) noexcept override {
    if (primary_->Owns(block)) {
      primary_->Deallocate(block);
      return;
    }
    fallback_->Deallocate(block);
  }

  bool Owns(const Block& block) const noexcept override {
    return primary_->Owns(block) || fallback_->Owns(block);
  }

 private:
  AllocatorImplPtr primary_;
  AllocatorImplPtr fallback_;
};

// TODO: batch allocation
class Pool : public AllocatorImpl {
 public:
  explicit Pool(AllocatorImplPtr allocator, size_t min_size, size_t max_size, unsigned pool_size)
      : allocator_(std::move(allocator)),
        min_size_(min_size),
        max_size_(max_size),
        pool_size_(pool_size) {
    free_.reserve(pool_size);
  }

  ~Pool() override {
    while (!free_.empty()) {
      Block block(free_.back(), max_size_);
      allocator_->Deallocate(block);
      free_.pop_back();
    }
  }

  Block Allocate(size_t size) noexcept override {
    if (min_size_ <= size && size <= max_size_) {
      if (!free_.empty()) {
        auto handle = free_.back();
        free_.pop_back();
        return Block{handle, max_size_};
      } else {
        return allocator_->Allocate(max_size_);
      }
    }
    return Block{};
  }

  void Deallocate(Block& block) noexcept override {
    if (Owns(block)) {
      if (free_.size() < pool_size_) {
        free_.push_back(block.handle);
        block.handle = nullptr;
        block.size = 0;
      } else {
        allocator_->Deallocate(block);
      }
    }
  }

  bool Owns(const Block& block) const noexcept override {
    return block.handle && min_size_ <= block.size && block.size <= max_size_;
  }

 private:
  AllocatorImplPtr allocator_;
  size_t min_size_;
  size_t max_size_;
  unsigned pool_size_;
  std::vector<void*> free_;
};

class Tree : public AllocatorImpl {
  static constexpr auto kQuantizer = 100;

 public:
  Tree(AllocatorImplPtr allocator, size_t max_bytes, float threshold)
      : allocator_(std::move(allocator)), max_tree_bytes_(max_bytes) {
    if (threshold) {
      thresh_numerator_ = static_cast<int>(threshold * kQuantizer);
      thresh_denominator_ = kQuantizer;
      auto divisor = std::gcd(thresh_numerator_, thresh_denominator_);
      thresh_numerator_ /= divisor;
      thresh_denominator_ /= divisor;
    }
  }

  ~Tree() override {
    for (const auto& [size, handle] : tree_) {
      Block block(handle, size);
      allocator_->Deallocate(block);
    }
  }

  Block Allocate(size_t size) noexcept override {
    if (auto it = tree_.lower_bound(size); it != tree_.end()) {
      if (size * thresh_denominator_ >= it->first * thresh_numerator_) {
        Block block(it->second, it->first);
        tree_bytes_ -= it->first;
        tree_.erase(it);
        return block;
      }
    }
    return allocator_->Allocate(size);
  }
  void Deallocate(Block& block) noexcept override {
    auto bytes = tree_bytes_ + block.size;
    if (bytes < max_tree_bytes_) {
      tree_.insert({block.size, block.handle});
      tree_bytes_ = bytes;
      block.size = 0;
      block.handle = nullptr;
    } else {
      allocator_->Deallocate(block);
    }
  }
  bool Owns(const Block& block) const noexcept override { return true; }

 private:
  AllocatorImplPtr allocator_;
  // threshold ~ thresh_numerator_ / thresh_denominator_
  int thresh_numerator_{};
  int thresh_denominator_{};
  std::multimap<size_t, void*> tree_;
  size_t max_tree_bytes_;
  size_t tree_bytes_{};
};

class Stats : public AllocatorImpl {
 public:
  explicit Stats(AllocatorImplPtr allocator, std::string name)
      : allocator_(std::move(allocator)), name_(std::move(name)) {}

  ~Stats() override {
    MMDEPLOY_INFO("=== {} ===", name_);
    MMDEPLOY_INFO("  Allocation: count={}, size={}MB, time={}ms", data_.allocation_count,
                  data_.allocated_bytes / (1024 * 1024.f),
                  static_cast<float>(data_.allocation_time));
    MMDEPLOY_INFO("Deallocation: count={}, size={}MB, time={}ms", data_.deallocation_count,
                  data_.deallocated_bytes / (1024 * 1024.f),
                  static_cast<float>(data_.deallocation_time));
    MMDEPLOY_INFO("Peak memory usage: size={}MB", data_.peak / (1024 * 1024.f));
  }

  Block Allocate(size_t size) noexcept override {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto block = allocator_->Allocate(size);
    auto t1 = std::chrono::high_resolution_clock::now();
    data_.allocation_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
    data_.allocated_bytes += block.size;
    data_.peak = std::max(data_.peak, data_.allocated_bytes - data_.deallocated_bytes);
    ++data_.allocation_count;
    return block;
  }

  void Deallocate(Block& block) noexcept override {
    ++data_.deallocation_count;
    data_.deallocated_bytes += block.size;
    auto t0 = std::chrono::high_resolution_clock::now();
    allocator_->Deallocate(block);
    auto t1 = std::chrono::high_resolution_clock::now();
    data_.deallocation_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  bool Owns(const Block& block) const noexcept override { return allocator_->Owns(block); }

  const char* Name() const noexcept override { return name_.c_str(); }

 private:
  struct Data {
    size_t allocation_count{};
    size_t deallocation_count{};
    size_t allocated_bytes{};
    size_t deallocated_bytes{};
    size_t peak{};
    double allocation_time{};
    double deallocation_time{};
  };
  Data data_;
  AllocatorImplPtr allocator_;
  std::string name_;
};

class Locked : public AllocatorImpl {
 public:
  explicit Locked(AllocatorImplPtr allocator) : allocator_(std::move(allocator)) {}
  Block Allocate(size_t size) noexcept override {
    std::lock_guard lock(mutex_);
    return allocator_->Allocate(size);
  }

  void Deallocate(Block& block) noexcept override {
    std::lock_guard lock(mutex_);
    allocator_->Deallocate(block);
  }

  bool Owns(const Block& block) const noexcept override {
    std::lock_guard lock(mutex_);
    return allocator_->Owns(block);
  }

 private:
  AllocatorImplPtr allocator_;
  mutable std::mutex mutex_;
};

class Segregator : public AllocatorImpl {
 public:
  Segregator(size_t threshold, AllocatorImplPtr small, AllocatorImplPtr large)
      : threshold_(threshold), small_(std::move(small)), large_(std::move(large)) {}

  Block Allocate(size_t size) noexcept override {
    if (size <= threshold_) {
      return small_->Allocate(size);
    }
    return large_->Allocate(size);
  }

  void Deallocate(Block& block) noexcept override {
    if (block.size <= threshold_) {
      return small_->Deallocate(block);
    }
    return large_->Deallocate(block);
  }

  bool Owns(const Block& block) const noexcept override {
    if (block.size <= threshold_) {
      return small_->Owns(block);
    }
    return large_->Owns(block);
  }

 private:
  size_t threshold_;
  AllocatorImplPtr small_;
  AllocatorImplPtr large_;
};

template <typename Allocator>
class AllocatorAdapter : public AllocatorImpl {
 public:
  Block Allocate(size_t size) noexcept override { return allocator_.Allocate(size); }
  void Deallocate(Block& block) noexcept override { return allocator_.Deallocate(block); }
  bool Owns(const Block& block) const noexcept override { return allocator_.Owns(block); }

 private:
  Allocator allocator_;
};

class Bucketizer : public AllocatorImpl {
 public:
  using AllocatorCreator = std::function<AllocatorImplPtr(size_t, size_t)>;
  Bucketizer(const AllocatorCreator& creator, size_t min_size, size_t max_size, size_t step_size)
      : min_size_(min_size), max_size_(max_size), step_size_(step_size) {
    for (auto base = min_size_; base < max_size_; base += step_size_) {
      //      MMDEPLOY_ERROR("{}, {}", base, base + step_size - 1);
      allocator_.push_back(creator(base, base + step_size - 1));
    }
    //    MMDEPLOY_ERROR("{}", allocator_.size());
  }

  Block Allocate(size_t size) noexcept override {
    auto index = (size - min_size_) / step_size_;
    if (0 <= index && index < allocator_.size()) {
      return allocator_[index]->Allocate(size);
    }
    return Block{};
  }

  void Deallocate(Block& block) noexcept override {
    auto index = (block.size - min_size_) / step_size_;
    if (0 <= index && index < allocator_.size()) {
      return allocator_[index]->Deallocate(block);
    }
  }

  bool Owns(const Block& block) const noexcept override {
    return min_size_ <= block.size && block.size < max_size_;
  }

 private:
  std::vector<AllocatorImplPtr> allocator_;
  size_t min_size_;
  size_t max_size_;
  size_t step_size_;
};

inline AllocatorImplPtr CreateFallback(AllocatorImplPtr primary, AllocatorImplPtr fallback) {
  return std::make_shared<Fallback>(std::move(primary), std::move(fallback));
}

inline AllocatorImplPtr CreateStats(const std::string& name, AllocatorImplPtr allocator) {
  return std::make_shared<Stats>(std::move(allocator), name);
}

inline AllocatorImplPtr CreatePool(size_t min_size, size_t max_size, unsigned int pool_size,
                                   AllocatorImplPtr allocator) {
  return std::make_shared<Pool>(std::move(allocator), min_size, max_size, pool_size);
}

inline AllocatorImplPtr CreateSegregator(size_t threshold, AllocatorImplPtr small,
                                         AllocatorImplPtr large) {
  return std::make_shared<Segregator>(threshold, std::move(small), std::move(large));
}

inline AllocatorImplPtr CreateBucketizer(size_t min_size, size_t max_size, size_t step_size,
                                         const Bucketizer::AllocatorCreator& creator) {
  return std::make_shared<Bucketizer>(creator, min_size, max_size, step_size);
}

inline AllocatorImplPtr CreatePoolBucketizer(size_t min_size, size_t max_size, size_t step_size,
                                             unsigned pool_size,
                                             const AllocatorImplPtr& allocator) {
  auto creator = [&](size_t lo, size_t hi) {
    return std::make_shared<Locked>(CreatePool(lo, hi, pool_size, allocator));
  };
  return CreateBucketizer(min_size, max_size, step_size, creator);
}

}  // namespace mmdeploy::framework::device_allocator

#endif  // MMDEPLOY_SRC_CORE_DEVICE_ALLOCATOR_H_
