// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_
#define MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_

#include <cstdint>

namespace mmdeploy::mmocr {

namespace dbnet {

void SigmoidAndThreshold(const float* d_data, int n, float thr, float* d_score, uint8_t* d_mask);

}

}  // namespace mmdeploy::mmocr

#endif  // MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_
