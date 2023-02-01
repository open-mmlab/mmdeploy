// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

#include "mmdeploy/operation/cuda/permute.h"

namespace mmdeploy {
namespace operation {
namespace cuda {
namespace impl {

template <typename T>
__global__ void permute(const T* src, const TensorStride src_strides, T* dst,
                        const TensorStride dst_strides, int ndim, int total) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= total) {
    return;
  }

  int remaining = u;
  int v = 0;
  for (size_t i = 0; i < ndim; i++) {
    int p = remaining / dst_strides.v_[i];
    remaining -= p * dst_strides.v_[i];
    v += p * src_strides.v_[i];
  }
  dst[u] = src[v];
}

template <typename T>
void Permute(const T* src, const TensorStride& src_strides, T* dst, const TensorStride& dst_strides,
             int ndim, int total, cudaStream_t stream) {
  int thread_num = 256;
  int block_num = (total + thread_num - 1) / thread_num;
  permute<T><<<block_num, thread_num, 0, stream>>>(src, src_strides, dst, dst_strides, ndim, total);
}

template void Permute<float>(const float* src, const TensorStride& src_strides, float* dst,
                             const TensorStride& dst_strides, int ndim, int total,
                             cudaStream_t stream);

template void Permute<uint8_t>(const uint8_t* src, const TensorStride& src_strides, uint8_t* dst,
                               const TensorStride& dst_strides, int ndim, int total,
                               cudaStream_t stream);

}  // namespace impl
}  // namespace cuda
}  // namespace operation
}  // namespace mmdeploy
