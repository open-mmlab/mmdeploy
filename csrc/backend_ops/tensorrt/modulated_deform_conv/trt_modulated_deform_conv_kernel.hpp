// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_MODULATED_DEFORM_CONV_KERNEL_HPP
#define TRT_MODULATED_DEFORM_CONV_KERNEL_HPP
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename scalar_t>
void ModulatedDeformConvForwardCUDAKernelLauncher(
    const scalar_t* input, const scalar_t* weight, const scalar_t* bias, const scalar_t* offset,
    const scalar_t* mask, scalar_t* output, void* workspace, int batch, int channels, int height,
    int width, int channels_out, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w,
    int pad_h, int dilation_w, int dilation_h, int group, int deformable_group, int im2col_step,
    cublasHandle_t cublas_handle, cudaStream_t stream);

#endif
