#ifndef TRT_BICUBIC_INTERPOLATE_KERNEL_HPP
#define TRT_BICUBIC_INTERPOLATE_KERNEL_HPP
#include <cuda_runtime.h>

#include "common_cuda_helper.hpp"

template <typename scalar_t>
void bicubic_interpolate(const scalar_t *input, scalar_t *output, int batch, int channels,
                         int in_height, int in_width, int out_height, int out_width,
                         bool align_corners, cudaStream_t stream);
#endif  // TRT_BICUBIC_INTERPOLATE_KERNEL_HPP
