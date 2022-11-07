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

template <typename T, typename... Args>
static unique_ptr<T> Create(const Device& device, Args&&... args) {
  auto platform = GetPlatformName(device);
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

struct Context {
  Device device;
  Stream stream;
};

class MMDEPLOY_API Session {
 public:
  Session();
  explicit Session(const Stream& stream);

  Session(const Session&) = delete;
  Session(Session&&) noexcept = delete;
  Session& operator=(const Session&) = delete;
  Session& operator=(Session&&) noexcept = delete;

  ~Session();

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

  template <typename T>
  Result<T> Track(Result<T> val) {
    if (val.has_value()) {
      return Track(val.value());
    } else {
      return val.as_failure();
    }
  }

  const std::vector<Buffer>& buffers() const noexcept { return buffers_; }

 private:
  Session* parent_;
  Stream stream_;
  std::vector<Buffer> buffers_;
};

MMDEPLOY_API Session& gSession();

class Operation {
 public:
  explicit Operation(Context context) : context_(std::move(context)) {}
  virtual ~Operation() = default;

  const Device& device() const noexcept { return context_.device; }
  Stream& stream() noexcept { return context_.stream; }

  template <typename T>
  T Secure(T&& val) {
    return (T &&) val;
  }

  Tensor Secure(Tensor val) {
    if (val.device() == device()) {
      return val;
    }

    TensorDesc desc{device(), val.data_type(), val.shape(), val.name()};
    Tensor dst(desc);

    Copy(val.buffer(), dst.buffer(), val.byte_size()).value();

    return gSession().track(dst);
  }

  Mat Secure(Mat val) {
    if (val.device() == device()) {
      return val;
    }

    Mat dst{val.height(), val.width(), val.pixel_format(), val.type(), device()};

    Copy(val.buffer(), dst.buffer(), val.byte_size()).value();

    return gSession().track(dst);
  }

  Result<void> Copy(const Buffer& src, Buffer& dst, size_t size) {
    OUTCOME_TRY(stream().Copy(src, dst, size));
    if (dst.GetDevice() != stream().GetDevice()) {
      OUTCOME_TRY(stream().Wait());
    }
    return success();
  }

 protected:
  Context context_;
};

template <typename Op, typename... Args>
auto apply(Op& op, Args&&... args) {
  return gSession().Track(op.apply(op.Secure((Args &&) args)...));
}

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
