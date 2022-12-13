// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_UTILS_DLPACK_DLPACK_UTILS_H_
#define MMDEPLOY_CSRC_UTILS_DLPACK_DLPACK_UTILS_H_

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/tensor.h"

struct DLManagedTensor;
namespace mmdeploy {

Result<DLManagedTensor*> ToDLPack(framework::Tensor& tensor, framework::Stream stream = {});
Result<framework::Tensor> FromDLPack(DLManagedTensor* managed_tensor, const std::string& name = "",
                                     framework::Stream stream = {});
}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_UTILS_DLPACK_DLPACK_UTILS_H_
