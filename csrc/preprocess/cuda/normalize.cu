// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace mmdeploy {
namespace cuda {

template <typename T, int channels>
__global__ void normalize(const T* src, int height, int width, int stride, float* output,
                          const float3 mean, const float3 std, bool to_rgb) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (x >= width || y >= height) {
    return;
  }

  int loc = y * stride + x * channels;
  auto mean_ptr = &mean.x;
  auto std_ptr = &std.x;
  if (to_rgb) {
    for (int c = 0; c < channels; ++c) {
      output[loc + c] = ((float)src[loc + channels - 1 - c] - mean_ptr[c]) * std_ptr[c];
    }
  } else {
    for (int c = 0; c < channels; ++c) {
      output[loc + c] = ((float)src[loc + c] - mean_ptr[c]) * std_ptr[c];
    }
  }
}

template <typename T, int channels>
void Normalize(const T* src, int height, int width, int stride, float* output, const float* mean,
               const float* std, bool to_rgb, cudaStream_t stream) {
  const dim3 thread_block(16, 16);
  const dim3 num_blocks((width + thread_block.x - 1) / thread_block.x,
                        (height + thread_block.y - 1) / thread_block.y);
  const float3 _mean{mean[0], mean[1], mean[2]};
  const float3 _std{float(1. / std[0]), float(1. / std[1]), float(1. / std[2])};
  normalize<T, channels><<<num_blocks, thread_block, 0, stream>>>(src, height, width, stride,
                                                                  output, _mean, _std, to_rgb);
}

template void Normalize<uint8_t, 3>(const uint8_t* src, int height, int width, int stride,
                                    float* output, const float* mean, const float* std, bool to_rgb,
                                    cudaStream_t stream);
template void Normalize<uint8_t, 1>(const uint8_t* src, int height, int width, int stride,
                                    float* output, const float* mean, const float* std, bool to_rgb,
                                    cudaStream_t stream);

template void Normalize<float, 3>(const float* src, int height, int width, int stride,
                                  float* output, const float* mean, const float* std, bool to_rgb,
                                  cudaStream_t stream);
template void Normalize<float, 1>(const float* src, int height, int width, int stride,
                                  float* output, const float* mean, const float* std, bool to_rgb,
                                  cudaStream_t stream);
}  // namespace cuda
}  // namespace mmdeploy
