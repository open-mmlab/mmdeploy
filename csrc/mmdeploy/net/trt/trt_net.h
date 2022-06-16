// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_TRT_TRT_NET_H_
#define MMDEPLOY_SRC_NET_TRT_TRT_NET_H_

#include "NvInferRuntime.h"
#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/net.h"

namespace mmdeploy {

namespace trt_detail {

template <typename T>
class TRTWrapper {
 public:
  TRTWrapper() : ptr_(nullptr) {}
  TRTWrapper(T* ptr) : ptr_(ptr) {}  // NOLINT
  ~TRTWrapper() { reset(); }
  TRTWrapper(const TRTWrapper&) = delete;
  TRTWrapper& operator=(const TRTWrapper&) = delete;
  TRTWrapper(TRTWrapper&& other) noexcept { *this = std::move(other); }
  TRTWrapper& operator=(TRTWrapper&& other) noexcept {
    reset(std::exchange(other.ptr_, nullptr));
    return *this;
  }
  T& operator*() { return *ptr_; }
  T* operator->() { return ptr_; }
  void reset(T* p = nullptr) {
    if (auto old = std::exchange(ptr_, p)) {  // NOLINT
#if NV_TENSORRT_MAJOR < 8
      old->destroy();
#else
      delete old;
#endif
    }
  }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

 private:
  T* ptr_;
};

// clang-format off
template <typename T>
explicit TRTWrapper(T*) -> TRTWrapper<T>;
// clang-format on
}  // namespace trt_detail

class TRTNet : public Net {
 public:
  ~TRTNet() override;
  Result<void> Init(const Value& cfg) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override;

 private:
 private:
  trt_detail::TRTWrapper<nvinfer1::ICudaEngine> engine_;
  trt_detail::TRTWrapper<nvinfer1::IExecutionContext> context_;
  std::vector<int> input_ids_;
  std::vector<int> output_ids_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  Device device_;
  Stream stream_;
  Event event_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_NET_TRT_TRT_NET_H_
