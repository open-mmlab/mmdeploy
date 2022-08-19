// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_UTILS_H
#define MMDEPLOY_TRANSFORM_UTILS_H

#include <utility>

#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"

namespace mmdeploy {
/**
 *
 * @param src
 * @param device
 * @param stream
 * @return
 */
MMDEPLOY_API Result<Mat> MakeAvailableOnDevice(const Mat& src, const Device& device,
                                               Stream& stream);

/**
 *
 * @param src
 * @param device
 * @param stream
 * @return
 */
MMDEPLOY_API Result<Tensor> MakeAvailableOnDevice(const Tensor& src, const Device& device,
                                                  Stream& stream);

// Calls stream.Wait() on destruction if active is true. This class is used to force a wait
// operation before intermediate variables goes out of scope. Add variables in consideration to the
// tailing parameter pack to ensure correctness (this make sure SyncOnScopeExit is created later
// (thus will be destructed earlier) than the variables

class MMDEPLOY_API SyncOnScopeExit {
 public:
  template <typename... Ts>
  explicit SyncOnScopeExit(Stream& stream, bool active, Ts&&...) noexcept
      : stream_(stream), active_(active) {}

  ~SyncOnScopeExit();

 private:
  bool active_;
  Stream& stream_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_TRANSFORM_UTILS_H
