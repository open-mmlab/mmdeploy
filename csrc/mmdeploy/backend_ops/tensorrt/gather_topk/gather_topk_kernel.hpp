// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_GRID_SAMPLER_KERNEL_HPP
#define TRT_GRID_SAMPLER_KERNEL_HPP
#include <cuda_runtime.h>

template <typename scalar_t>
void gather_topk_impl(const scalar_t* input, const int* indices, const int* dims, int nbDims,
                      const int* indices_dims, int indice_nbDims, scalar_t* output,
                      cudaStream_t stream);
#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
