// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_MS_DEFORM_ATTN_KERNEL_HPP
#define TRT_MS_DEFORM_ATTN_KERNEL_HPP
#include <cublas_v2.h>
#include <cuda_runtime.h>

int32_t ms_deform_attn_cuda_forward(const float* value, const int32_t* spatialShapes,
    const int32_t* levelStartIndex, const float* samplingLoc, const float* attnWeight, float* output, int32_t batch,
    int32_t mSpatialSize, int32_t mNumHeads, int32_t mChannels, int32_t mNumLevels, int32_t mNumQuery, int32_t mNumPoint,
    cudaStream_t stream);

int32_t ms_deform_attn_cuda_forward(const __half* value, int32_t const* spatialShapes,
    int32_t const* levelStartIndex, const __half* samplingLoc, const __half* attnWeight, __half* output, int32_t batch,
    int32_t mSpatialSize, int32_t mNumHeads, int32_t mChannels, int32_t mNumLevels, int32_t mNumQuery, int32_t mNumPoint,
    cudaStream_t stream);

#endif
