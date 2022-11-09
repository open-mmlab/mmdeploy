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

struct Context {
  Device device;
  Stream stream;
};

class ContextGuard {
 public:
  explicit ContextGuard(Context context);
  ~ContextGuard();

  ContextGuard(const ContextGuard&) = delete;
  ContextGuard(ContextGuard&&) noexcept = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;
  ContextGuard& operator=(ContextGuard&&) noexcept = delete;

 private:
  Context context_;
  Context* parent_;
};

MMDEPLOY_API Context& gContext();

template <typename T, typename... Args>
static unique_ptr<T> Create(Args&&... args) {
  auto platform = GetPlatformName(gContext().device);
  assert(platform);
  std::vector<string_view> candidates{platform, "cpu"};
  if (candidates[0] == candidates[1]) {
    candidates.pop_back();
  }
  for (const auto& name : candidates) {
    if (auto creator = gRegistry<T>().Get(name)) {
      return creator->Create((Args &&) args...);
    }
  }
  return nullptr;
}

class MMDEPLOY_API Session {
 public:
  Session();
  explicit Session(const Stream& stream);
  ~Session();

  Session(const Session&) = delete;
  Session(Session&&) noexcept = delete;
  Session& operator=(const Session&) = delete;
  Session& operator=(Session&&) noexcept = delete;

  Tensor& track(Tensor& tensor) {
    buffers_.push_back(tensor.buffer());
    return tensor;
  }

  Mat& track(Mat& mat) {
    buffers_.push_back(mat.buffer());
    return mat;
  }

  Buffer& track(Buffer& buffer) {
    buffers_.push_back(buffer);
    return buffer;
  }

  template <typename T, typename... Args>
  T Create(Args&&... args) {
    return Track(T((Args &&) args...));
  }

  template <typename T>
  T Track(T val) {
    return val;
  }

  Mat Track(Mat mat) { return track(mat); }

  Tensor Track(Tensor tensor) { return track(tensor); }

  const std::vector<Buffer>& buffers() const noexcept { return buffers_; }

 private:
  Session* parent_;
  Stream stream_;
  std::vector<Buffer> buffers_;
};

MMDEPLOY_API Session& gSession();

class Operation {
 public:
  Operation() : context_(gContext()) {}
  explicit Operation(Context context) : context_(std::move(context)) {}
  virtual ~Operation() = default;

  const Device& device() const noexcept { return context_.device; }
  Stream& stream() noexcept { return context_.stream; }

 protected:
  Context context_;
};

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
