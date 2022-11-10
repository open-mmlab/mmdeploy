// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
#define MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"

namespace mmdeploy::operation {

using namespace mmdeploy::framework;
using std::string_view;
using std::unique_ptr;

class Context {
 public:
  explicit Context(Device device);
  explicit Context(Stream stream);
  explicit Context(Device device, Stream stream);
  ~Context();

  Context(const Context&) = delete;
  Context(Context&&) noexcept = delete;
  Context& operator=(const Context&) = delete;
  Context& operator=(Context&&) noexcept = delete;

  void Track(const Tensor& tensor) { buffers_.push_back(tensor.buffer()); }
  void Track(const Mat& mat) { buffers_.push_back(mat.buffer()); };
  void Track(const Buffer& buffer) { buffers_.push_back(buffer); };

  template <typename T, typename... Args>
  T Create(Args&&... args) {
    return _track(T((Args &&) args...));
  }

  const Device& device() const noexcept { return device_; }
  Stream& stream() noexcept { return stream_; }
  const std::vector<Buffer>& buffers() const noexcept { return buffers_; }

 private:
  Tensor&& _track(Tensor&& tensor) {
    Track(tensor);
    return std::move(tensor);
  }
  Mat&& _track(Mat&& mat) {
    Track(mat);
    return std::move(mat);
  }
  Buffer&& _track(Buffer&& buffer) {
    Track(buffer);
    return std::move(buffer);
  }

 private:
  Device device_;
  Stream stream_;
  std::vector<Buffer> buffers_;
  Context* parent_;
};

MMDEPLOY_API Context& gContext();

template <typename T, typename... Args>
static unique_ptr<T> Create(Args&&... args) {
  std::vector<Device> candidates{gContext().device()};
  if (candidates[0].is_device()) {
    candidates.emplace_back(0);
  }
  for (const auto& device : candidates) {
    if (auto platform = GetPlatformName(device)) {
      if (auto creator = gRegistry<T>().Get(platform)) {
        Context context(device);
        return creator->Create((Args &&) args...);
      }
    }
  }
  return nullptr;
}

class Operation {
 public:
  Operation() : device_(gContext().device()) {}
  virtual ~Operation() = default;

  const Device& device() const noexcept { return device_; }
  static Stream& stream() noexcept { return gContext().stream(); }

 protected:
  Device device_;
};

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
