// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

namespace mmdeploy {
namespace operation {
namespace cuda {

template <typename From, typename To>
__global__ void _Cast(const From* src, To* dst, size_t n) {
  auto idx = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
    dst[i] = static_cast<To>(src[i]);
  }
}

template <typename From, typename To>
void Cast(const From* src, To* dst, size_t n, cudaStream_t stream) {
  size_t n_threads = 256;
  size_t n_blocks = (n + n_threads - 1) / n_threads;
  _Cast<<<n_blocks, n_threads, 0, stream>>>(src, dst, n);
}

template void Cast(const uint8_t*, float*, size_t, cudaStream_t);

}  // namespace cuda
}  // namespace operation
}  // namespace mmdeploy
