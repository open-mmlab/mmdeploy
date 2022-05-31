// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_
#define MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_

#include <cstdint>

#include "cuda_runtime.h"

namespace mmdeploy {

namespace mmocr {

namespace panet {

void ProcessMasks(const float* d_text_pred, const float* d_kernel_pred, float text_thr,
                  float kernel_thr, int n, uint8_t* d_text_mask, uint8_t* d_kernel_mask,
                  float* d_text_score, cudaStream_t stream);

void Transpose(const float* d_input, int h, int w, float* d_output, cudaStream_t stream);

}  // namespace panet

namespace dbnet {

void Threshold(const float* d_score, int n, float thr, uint8_t* d_mask, cudaStream_t stream);

}

namespace psenet {

void ProcessMasks(const float* d_preds, int c, int n, float thr, uint8_t* d_masks, float* d_score,
                  cudaStream_t stream);

}

}  // namespace mmocr

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_CODEBASE_MMOCR_CUDA_UTILS_H_
