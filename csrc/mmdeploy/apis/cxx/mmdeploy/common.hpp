// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_COMMON_H_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_COMMON_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "mmdeploy/common.h"
#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/types.h"
#include "mmdeploy/executor.h"
#include "mmdeploy/model.h"

#ifndef MMDEPLOY_CXX_USE_OPENCV
#define MMDEPLOY_CXX_USE_OPENCV 1
#endif

#if MMDEPLOY_CXX_USE_OPENCV
#include "opencv2/core/core.hpp"
#endif

namespace mmdeploy {

namespace cxx {

using Rect = mmdeploy_rect_t;

template <typename T>
class UniqueHandle : public NonCopyable {
 public:
  UniqueHandle() = default;
  explicit UniqueHandle(T handle) : handle_(handle) {}

  // derived class must destroy the object and reset `handle_`
  ~UniqueHandle() { assert(handle_ == nullptr); }

  UniqueHandle(UniqueHandle&& o) noexcept : handle_(std::exchange(o.handle_, nullptr)) {}
  UniqueHandle& operator=(UniqueHandle&& o) noexcept {
    if (this != &o) {
      handle_ = std::exchange(o.handle_, nullptr);
    }
    return *this;
  }

  explicit operator T() const noexcept { return handle_; }
  T operator->() const noexcept { return handle_; }

 protected:
  T handle_{};
};

class Model {
 public:
  explicit Model(const char* path) {
    mmdeploy_model_t model{};
    auto ec = mmdeploy_model_create_by_path(path, &model);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
    model_.reset(model, [](auto p) { mmdeploy_model_destroy(p); });
  }

  explicit Model(const std::string& path) : Model(path.c_str()) {}

  Model(const void* buffer, size_t size) {
    mmdeploy_model_t model{};
    auto ec = mmdeploy_model_create(buffer, static_cast<int>(size), &model);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
    model_.reset(model, [](auto p) { mmdeploy_model_destroy(p); });
  }

  operator mmdeploy_model_t() const noexcept { return model_.get(); }

 private:
  std::shared_ptr<mmdeploy_model> model_{};
};

class Device {
 public:
  explicit Device(std::string name, int index = 0) : name_(std::move(name)), index_(index) {
    mmdeploy_device_t device{};
    auto ec = mmdeploy_device_create(name_.c_str(), index, &device);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
    device_.reset(device, [](auto p) { mmdeploy_device_destroy(p); });
  }

  const char* name() const noexcept { return name_.c_str(); }
  int index() const noexcept { return index_; }

  operator mmdeploy_device_t() const noexcept { return device_.get(); }

 private:
  std::string name_;
  int index_;
  std::shared_ptr<mmdeploy_device> device_;
};

class Profiler {
 public:
  explicit Profiler(std::string_view path) : path_(path) {
    mmdeploy_profiler_t profiler{};
    auto ec = mmdeploy_profiler_create(path_.c_str(), &profiler);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
    profiler_.reset(profiler, [](auto p) { mmdeploy_profiler_destroy(p); });
  };

  operator mmdeploy_profiler_t() const noexcept { return profiler_.get(); }

 private:
  std::string path_;
  std::shared_ptr<mmdeploy_profiler> profiler_;
};

class Mat {
 public:
  Mat() : desc_{} {}

  Mat(int height, int width, int channels, mmdeploy_pixel_format_t format,
      mmdeploy_data_type_t type, uint8_t* data, mmdeploy_device_t device = nullptr)
      : desc_{data, height, width, channels, format, type, device} {}

  Mat(const mmdeploy_mat_t& desc) : desc_(desc) {}  // NOLINT

  const mmdeploy_mat_t& desc() const noexcept { return desc_; }

#if MMDEPLOY_CXX_USE_OPENCV
  Mat(const cv::Mat& mat, mmdeploy_pixel_format_t pixel_format)
      : desc_{mat.data, mat.rows, mat.cols, mat.channels(), pixel_format, GetCvType(mat.depth())} {
    if (pixel_format == MMDEPLOY_PIXEL_FORMAT_COUNT) {
      throw_exception(eNotSupported);
    }
    if (desc_.type == MMDEPLOY_DATA_TYPE_COUNT) {
      throw_exception(eNotSupported);
    }
  }
  Mat(const cv::Mat& mat) : Mat(mat, GetCvFormat(mat.channels())) {}

  static mmdeploy_data_type_t GetCvType(int depth) {
    switch (depth) {
      case CV_8U:
        return MMDEPLOY_DATA_TYPE_UINT8;
      case CV_32F:
        return MMDEPLOY_DATA_TYPE_FLOAT;
      default:
        return MMDEPLOY_DATA_TYPE_COUNT;
    }
  }
  static mmdeploy_pixel_format_t GetCvFormat(int channels) {
    switch (channels) {
      case 1:
        return MMDEPLOY_PIXEL_FORMAT_GRAYSCALE;
      case 3:
        return MMDEPLOY_PIXEL_FORMAT_BGR;
      case 4:
        return MMDEPLOY_PIXEL_FORMAT_BGRA;
      default:
        return MMDEPLOY_PIXEL_FORMAT_COUNT;
    }
  }
#endif
 private:
  mmdeploy_mat_t desc_;
};

template <typename T>
class Result_ {
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = T*;

  Result_(size_t offset, size_t size, std::shared_ptr<T> data)
      : offset_(offset), size_(size), data_(std::move(data)) {}

  T& operator[](size_t index) const noexcept { return *(data_.get() + offset_ + index); }
  size_t size() const noexcept { return size_; }
  T* begin() const noexcept { return data_.get() + offset_; }
  T* end() const noexcept { return begin() + size_; }

  T* operator->() const noexcept { return data_.get(); }
  T& operator*() const noexcept { return *data_; }

 private:
  size_t offset_;
  size_t size_;
  std::shared_ptr<T> data_;
};

inline const mmdeploy_mat_t* reinterpret(const Mat* p) {
  return reinterpret_cast<const mmdeploy_mat_t*>(p);
}

class Scheduler {
 public:
  explicit Scheduler(mmdeploy_scheduler_t scheduler) {
    scheduler_.reset(scheduler, [](auto p) { mmdeploy_scheduler_destroy(p); });
  }

  static Scheduler ThreadPool(int num_threads) {
    return Scheduler(mmdeploy_executor_create_thread_pool(num_threads));
  }
  static Scheduler Thread() { return Scheduler(mmdeploy_executor_create_thread()); }

  operator mmdeploy_scheduler_t() const noexcept { return scheduler_.get(); }

 private:
  std::shared_ptr<mmdeploy_scheduler> scheduler_;
};

class Context {
 public:
  Context() {
    mmdeploy_context_t context{};
    mmdeploy_context_create(&context);
    context_.reset(context, [](auto p) { mmdeploy_context_destroy(p); });
  }
  /* implicit */ Context(const Device& device) : Context() { Add(device); }

  void Add(const std::string& name, const Scheduler& scheduler) {
    mmdeploy_context_add(*this, MMDEPLOY_TYPE_SCHEDULER, name.c_str(), scheduler);
  }

  void Add(const std::string& name, const Model& model) {
    mmdeploy_context_add(*this, MMDEPLOY_TYPE_MODEL, name.c_str(), model);
  }

  void Add(const Device& device) {
    mmdeploy_context_add(*this, MMDEPLOY_TYPE_DEVICE, nullptr, device);
  }

  void Add(const Profiler& profiler) {
    mmdeploy_context_add(*this, MMDEPLOY_TYPE_PROFILER, nullptr, profiler);
  }

  operator mmdeploy_context_t() const noexcept { return context_.get(); }

 private:
  std::shared_ptr<mmdeploy_context> context_;
};

}  // namespace cxx

using cxx::Context;
using cxx::Device;
using cxx::Mat;
using cxx::Model;
using cxx::Profiler;
using cxx::Rect;
using cxx::Scheduler;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_COMMON_H_
