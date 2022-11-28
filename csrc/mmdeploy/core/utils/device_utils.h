// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_UTILS_H
#define MMDEPLOY_TRANSFORM_UTILS_H

#include <utility>

#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"

namespace mmdeploy::framework {
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

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_TRANSFORM_UTILS_H
