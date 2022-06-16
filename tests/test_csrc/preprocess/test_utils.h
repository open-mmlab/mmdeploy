// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TEST_TRANSFORM_UTILS_H
#define MMDEPLOY_TEST_TRANSFORM_UTILS_H

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::test {
std::unique_ptr<Transform> CreateTransform(const Value& cfg, Device device, Stream stream);

std::vector<int64_t> Shape(const Value& value, const std::string& shape_key);

std::vector<float> ImageNormCfg(const Value& value, const std::string& key);

}  // namespace mmdeploy::test

#endif  // MMDEPLOY_TEST_TRANSFORM_UTILS_H
