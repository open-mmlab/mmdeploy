#ifndef TRT_DEFORM_CONV_V3_KERNEL_HPP
#define TRT_DEFORM_CONV_V3_KERNEL_HPP
#include <cuda_runtime.h>

#include "common_cuda_helper.hpp"

template <typename scalar_t>
void DeformConvv3ForwardCUDAKernelLauncher(
    const scalar_t* input, const scalar_t* offset, const scalar_t* mask, scalar_t* output,
    void* workspace, int batch, int channels, int height, int width, int channels_out, int kernel_w,
    int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h, int dilation_w, int dilation_h,
    int group, int group_channel, float offset_scale, int im2col_step, cudaStream_t stream);

#endif  // TRT_DEFORM_CONV_V3_KERNEL_HPP
