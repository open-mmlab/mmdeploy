//
// Created by zhangli on 11/3/22.
//

#ifndef MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
#define MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_

#include "mmdeploy/core/device.h"
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

class Session;

thread_local Session* g_session;

class Session {
 public:
  Session() : parent_(std::exchange(g_session, this)) {}

  ~Session() { g_session = std::exchange(parent_, nullptr); }

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

 private:
  Session* parent_;
  std::vector<Buffer> buffers_;
};

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
    if (val.device() == context_.device) {
      return val;
    }

    TensorDesc desc{context_.device, val.data_type(), val.shape(), val.name()};
    Tensor dst(desc);

    context_.stream.Copy(val.buffer(), dst.buffer(), val.byte_size()).value();
    MaybeSync();

    return g_session->track(dst);
  }

  Mat Secure(Mat val) {
    if (val.device() == context_.device) {
      return val;
    }

    Mat dst{val.height(), val.width(), val.pixel_format(), val.type(), context_.device};

    context_.stream.Copy(val.buffer(), dst.buffer(), val.byte_size()).value();
    MaybeSync();

    return g_session->track(dst);
  }

  void MaybeSync() {
    // ! When the target device is different from stream's device (e.g. DtoH), insert a sync op as
    //   on dst won't be synchronized wrt stream
    if (context_.device != context_.stream.GetDevice()) {
      context_.stream.Wait().value();
    }
  }

  template <typename T>
  T Track(T val) {
    return val;
  }

  Mat Track(Mat mat) { return g_session->track(mat); }

  Tensor Track(Tensor tensor) { return g_session->track(tensor); }

  template <typename T>
  Result<T> Track(Result<T> val) {
    if (val.has_value()) {
      return Track(val.value());
    } else {
      return val.as_failure();
    }
  }

 protected:
  Context context_;
};

template <typename T>
class CRTPOp : public Operation {
 public:
  using Operation::Operation;

  template <typename... Args>
  auto Apply(Args&&... args) {
    return Track(static_cast<T*>(this)->apply(Secure((Args &&) args)...));
  }
};

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
