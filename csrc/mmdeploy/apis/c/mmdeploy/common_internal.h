// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_APIS_C_COMMON_INTERNAL_H_
#define MMDEPLOY_CSRC_APIS_C_COMMON_INTERNAL_H_

#include "common.h"
#include "handle.h"
#include "mmdeploy/core/value.h"
#include "model.h"
#include "pipeline.h"

using namespace mmdeploy;

namespace {

inline mmdeploy_value_t Cast(Value* s) { return reinterpret_cast<mmdeploy_value_t>(s); }

inline Value* Cast(mmdeploy_value_t s) { return reinterpret_cast<Value*>(s); }

inline Value Take(mmdeploy_value_t v) {
  auto value = std::move(*Cast(v));
  mmdeploy_value_destroy(v);
  return value;
}

inline Value* Cast(mmdeploy_context_t c) { return reinterpret_cast<Value*>(c); }

mmdeploy_value_t Take(Value v) {
  return Cast(new Value(std::move(v)));  // NOLINT
}

mmdeploy_pipeline_t Cast(AsyncHandle* pipeline) {
  return reinterpret_cast<mmdeploy_pipeline_t>(pipeline);
}

AsyncHandle* Cast(mmdeploy_pipeline_t pipeline) { return reinterpret_cast<AsyncHandle*>(pipeline); }

mmdeploy_model_t Cast(Model* model) { return reinterpret_cast<mmdeploy_model_t>(model); }

Model* Cast(mmdeploy_model_t model) { return reinterpret_cast<Model*>(model); }

template <typename F>
std::invoke_result_t<F> Guard(F f) {
  try {
    return f();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return nullptr;
}

template <typename T, typename SFINAE = void>
class wrapped {};

template <typename T>
class wrapped<T, std::void_t<decltype(Cast(T{}))>> {
 public:
  wrapped() noexcept : v_(nullptr) {}
  explicit wrapped(T v) noexcept : v_(v) {}

  void reset() {
    if (v_) {
      delete Cast(v_);
      v_ = nullptr;
    }
  }

  ~wrapped() { reset(); }

  wrapped(const wrapped&) = delete;
  wrapped& operator=(const wrapped&) = delete;

  wrapped(wrapped&& other) noexcept : v_(other.release()) {}
  wrapped& operator=(wrapped&& other) noexcept {
    reset();
    v_ = other.release();
    return *this;
  }

  T release() noexcept { return std::exchange(v_, nullptr); }

  auto operator*() { return Cast(v_); }
  auto operator-> () { return Cast(v_); }

  T* ptr() noexcept { return &v_; }

  operator T() const noexcept { return v_; }  // NOLINT

 private:
  T v_;
};

}  // namespace

#endif  // MMDEPLOY_CSRC_APIS_C_COMMON_INTERNAL_H_
