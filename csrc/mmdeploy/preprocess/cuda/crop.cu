// Copyright (c) OpenMMLab. All rights reserved.

#include <stdint.h>

namespace mmdeploy {
namespace cuda {

template <typename T, int channels>
__global__ void crop(const T *src, int src_w, T *dst, int dst_h, int dst_w, int offset_h,
                     int offset_w) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst_w || y >= dst_h) return;
  int src_x = x + offset_w;
  int src_y = y + offset_h;

  int dst_loc = (y * dst_w + x) * channels;
  int src_loc = (src_y * src_w + src_x) * channels;

  for (int i = 0; i < channels; ++i) {
    dst[dst_loc + i] = src[src_loc + i];
  }
}

template <typename T, int channels>
void Crop(const T *src, int src_w, T *dst, int dst_h, int dst_w, int offset_h, int offset_w,
          cudaStream_t stream) {
  const dim3 thread_block(32, 8);
  const dim3 block_num((dst_w + thread_block.x - 1) / thread_block.x,
                       (dst_h + thread_block.y - 1) / thread_block.y);
  crop<T, channels>
      <<<block_num, thread_block, 0, stream>>>(src, src_w, dst, dst_h, dst_w, offset_h, offset_w);
}

template void Crop<uint8_t, 3>(const uint8_t *src, int src_w, uint8_t *dst, int dst_h, int dst_w,
                               int offset_h, int offset_w, cudaStream_t stream);

template void Crop<uint8_t, 1>(const uint8_t *src, int src_w, uint8_t *dst, int dst_h, int dst_w,
                               int offset_h, int offset_w, cudaStream_t stream);

template void Crop<float, 3>(const float *src, int src_w, float *dst, int dst_h, int dst_w,
                             int offset_h, int offset_w, cudaStream_t stream);

template void Crop<float, 1>(const float *src, int src_w, float *dst, int dst_h, int dst_w,
                             int offset_h, int offset_w, cudaStream_t stream);

}  // namespace cuda
}  // namespace mmdeploy
