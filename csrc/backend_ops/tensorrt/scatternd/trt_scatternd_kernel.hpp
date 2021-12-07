// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_SCATTERND_KERNEL_HPP
#define TRT_SCATTERND_KERNEL_HPP
#include <cuda_runtime.h>

template <typename T>
void TRTONNXScatterNDKernelLauncher(const T* data, const int* indices, const T* update,
                                    const int* dims, int nbDims, const int* indices_dims,
                                    int indice_nbDims, T* output, cudaStream_t stream);

#endif  // TRT_SCATTERND_KERNEL_HPP
