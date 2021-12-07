// Copyright (c) OpenMMLab. All rights reserved.
#ifndef ROI_ALIGN_CUDA_KERNEL_HPP
#define ROI_ALIGN_CUDA_KERNEL_HPP

#include "common_cuda_helper.hpp"

template <typename scalar_t>
void TRTRoIAlignForwardCUDAKernelLauncher(const scalar_t* input, const scalar_t* rois,
                                          scalar_t* output, scalar_t* argmax_y, scalar_t* argmax_x,
                                          int output_size, int channels, int height, int width,
                                          int aligned_height, int aligned_width,
                                          scalar_t spatial_scale, int sampling_ratio, int pool_mode,
                                          bool aligned, cudaStream_t stream);

#endif  // ROI_ALIGN_CUDA_KERNEL_HPP
