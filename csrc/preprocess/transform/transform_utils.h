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
Result<Mat> MakeAvailableOnDevice(const Mat& src, const Device& device, Stream& stream);

/**
 *
 * @param src
 * @param device
 * @param stream
 * @return
 */
Result<Tensor> MakeAvailableOnDevice(const Tensor& src, const Device& device, Stream& stream);
}  // namespace mmdeploy

#endif  // MMDEPLOY_TRANSFORM_UTILS_H
