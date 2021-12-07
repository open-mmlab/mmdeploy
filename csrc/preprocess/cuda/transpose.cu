// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

namespace mmdeploy {
namespace cuda {

template <typename T>
__global__ void transpose(const T* src, int height, int width, int channels, int src_width_stride,
                          T* dst, int dst_channel_stride) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  for (auto c = 0; c < channels; ++c) {
    dst[c * dst_channel_stride + y * width + x] = src[y * src_width_stride + x * channels + c];
  }
}

template <typename T>
void Transpose(const T* src, int height, int width, int channels, T* dst, cudaStream_t stream) {
  const dim3 thread_block(32, 8);
  const dim3 block_num((width + thread_block.x - 1) / thread_block.x,
                       (height + thread_block.y - 1) / thread_block.y);

  auto src_width_stride = width * channels;
  auto dst_channel_stride = width * height;

  transpose<T><<<block_num, thread_block, 0, stream>>>(src, height, width, channels,
                                                       src_width_stride, dst, dst_channel_stride);
}

template void Transpose<uint8_t>(const uint8_t* src, int height, int width, int channels,
                                 uint8_t* dst, cudaStream_t stream);

template void Transpose<float>(const float* src, int height, int width, int channels, float* dst,
                               cudaStream_t stream);
}  // namespace cuda
}  // namespace mmdeploy
