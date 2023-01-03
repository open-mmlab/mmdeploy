// Copyright (c) OpenMMLab. All rights reserved.

#include <stdint.h>
#include <stdio.h>

namespace mmdeploy {
namespace mmaction {
namespace cuda {

template <typename T>
__global__ void transpose(const T* src, const int* src_strides, T* dst, const int* dst_strides,
                          int ndim, int total) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= total) {
    return;
  }

  int remaining = u;
  int v = 0;
  for (int i = 0; i < ndim; i++) {
    int p = remaining / dst_strides[i];
    remaining -= p * dst_strides[i];
    v += p * src_strides[i];
  }
  dst[u] = src[v];
}

template <typename T>
void Transpose(const T* src, const int* src_strides, T* dst, const int* dst_strides, int ndim,
               int total, cudaStream_t stream) {
  int thread_num = 256;
  int block_num = (total + thread_num - 1) / thread_num;
  transpose<T>
      <<<block_num, thread_num, 0, stream>>>(src, src_strides, dst, dst_strides, ndim, total);
}

template void Transpose<float>(const float* src, const int* src_strides, float* dst,
                               const int* dst_strides, int ndim, int total, cudaStream_t stream);

}  // namespace cuda
}  // namespace mmaction
}  // namespace mmdeploy
