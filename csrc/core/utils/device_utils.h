// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_UTILS_H
#define MMDEPLOY_TRANSFORM_UTILS_H

#include <utility>

#include "core/mat.h"
#include "core/tensor.h"

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

class SyncOnScopeExit {
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
