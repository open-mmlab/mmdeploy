// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_UTILS_H
#define MMDEPLOY_TRANSFORM_UTILS_H

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

class ForceSync {
 public:
  template <typename...Ts>
  explicit ForceSync(Stream& stream, Ts&...) noexcept: stream_(stream) {}

  void set_active(bool active) noexcept { active_ = active; }
  
  ~ForceSync();
 private:
  bool active_ = true;
  Stream& stream_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_TRANSFORM_UTILS_H
