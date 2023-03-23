// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_MS_DEFORM_ATTN_KERNEL_HPP
#define TRT_MS_DEFORM_ATTN_KERNEL_HPP
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename scalar_t>
int32_t ms_deform_attn_cuda_forward(const scalar_t* value, const int32_t* spatialShapes,
                                    const int32_t* levelStartIndex, const scalar_t* samplingLoc,
                                    const scalar_t* attnWeight, scalar_t* output, int32_t batch,
                                    int32_t mSpatialSize, int32_t mNumHeads, int32_t mChannels,
                                    int32_t mNumLevels, int32_t mNumQuery, int32_t mNumPoint,
                                    cudaStream_t stream);

#endif
