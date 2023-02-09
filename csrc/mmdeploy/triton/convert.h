// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CONVERT_H
#define MMDEPLOY_CONVERT_H

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/value.h"

namespace triton::backend::mmdeploy {

std::vector<std::vector<::mmdeploy::framework::Tensor>> ConvertOutputToTensors(
    const std::string& type, int32_t request_count, const ::mmdeploy::Value& output,
    std::vector<std::string>& strings);

}

#endif  // MMDEPLOY_CONVERT_H
