// Copyright (c) OpenMMLab. All rights reserved.
#ifndef COMMON_CUDA_HELPER
#define COMMON_CUDA_HELPER

#include <cublas_v2.h>
#include <cuda.h>

#include <algorithm>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
inline int GET_BLOCKS(const int N) {
  int optimal_block_num = DIVUP(N, THREADS_PER_BLOCK);
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

/**
 * Returns a view of the original tensor with its dimensions permuted.
 *
 * @param[out] dst pointer to the destination tensor
 * @param[in] src pointer to the source tensor
 * @param[in] src_size shape of the src tensor
 * @param[in] permute The desired ordering of dimensions
 * @param[in] src_dim dim of src tensor
 * @param[in] stream cuda stream handle
 */
template <class scalar_t>
void memcpyPermute(scalar_t* dst, const scalar_t* src, int* src_size, int* permute, int src_dim,
                   cudaStream_t stream = 0);

template <typename scalar_t>
cublasStatus_t cublasGemmWrap(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k, const scalar_t* alpha,
                              const scalar_t* A, int lda, const scalar_t* B, int ldb,
                              const scalar_t* beta, scalar_t* C, int ldc);

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t* input, const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  scalar_t v1 = input[y_low * width + x_low];
  scalar_t v2 = input[y_low * width + x_high];
  scalar_t v3 = input[y_high * width + x_low];
  scalar_t v4 = input[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

#endif  // COMMON_CUDA_HELPER
