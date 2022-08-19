// Copyright (c) OpenMMLab. All rights reserved.

#include <stdint.h>

namespace mmdeploy {
namespace cuda {

template <int channels>
__global__ void cast(const uint8_t *src, int height, int width, float *dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int loc = (y * width + x) * channels;
  for (int i = 0; i < channels; ++i) {
    dst[loc + i] = src[loc + i];
  }
}

template <int channels>
void CastToFloat(const uint8_t *src, int height, int width, float *dst, cudaStream_t stream) {
  const dim3 thread_block(32, 8);
  const dim3 block_num((width + thread_block.x - 1) / thread_block.x,
                       (height + thread_block.y - 1) / thread_block.y);
  cast<channels><<<block_num, thread_block, 0, stream>>>(src, height, width, dst);
}

template void CastToFloat<3>(const uint8_t *src, int height, int width, float *dst,
                             cudaStream_t stream);

template void CastToFloat<1>(const uint8_t *src, int height, int width, float *dst,
                             cudaStream_t stream);

}  // namespace cuda
}  // namespace mmdeploy
