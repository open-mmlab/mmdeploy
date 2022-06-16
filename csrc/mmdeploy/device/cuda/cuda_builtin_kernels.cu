// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

namespace mmdeploy {
namespace cuda {

__global__ void FillKernel(void* dst, size_t dst_size, const void* pattern, size_t pattern_size) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  auto p_dst = static_cast<uchar1*>(dst);
  auto p_pattern = static_cast<const uchar1*>(pattern);

  for (; idx < dst_size; idx += blockDim.x * gridDim.x) {
    auto ptr = idx % pattern_size;
    p_dst[idx] = p_pattern[ptr];
  }
}

int Fill(void* dst, size_t dst_size, const void* pattern, size_t pattern_size,
         cudaStream_t stream) {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks = (dst_size + n_threads - 1) / n_threads;

  FillKernel<<<n_blocks, n_threads, 0, stream>>>(dst, dst_size, pattern, pattern_size);

  return 0;
}

}  // namespace cuda
}  // namespace mmdeploy
