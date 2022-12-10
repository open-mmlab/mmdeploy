// Copyright (c) OpenMMLab. All rights reserved.
#include "common_cuda_helper.hpp"
#include <onnxruntime_cxx_api.h>
#include <cuda_fp16.h>

namespace mmdeploy {
  const int MAXTENSORDIMS = 10;

  struct TensorDesc {
    int shape[MAXTENSORDIMS];
    int stride[MAXTENSORDIMS];
    int dim;
  };
}

using mmdeploy::TensorDesc;

template <class scalar_t>
__global__ void copy_permute_kernel(scalar_t *__restrict__ dst, const scalar_t *__restrict__ src,
                                    int n, TensorDesc ts_src_stride, TensorDesc ts_dst_stride,
                                    TensorDesc ts_permute) {
  const int src_dim = ts_src_stride.dim;
  const auto src_stride = ts_src_stride.stride;
  const auto dst_stride = ts_dst_stride.stride;
  const auto permute = ts_permute.shape;
  CUDA_1D_KERNEL_LOOP(index, n) {
    size_t dst_index = index;
    size_t src_index = 0;
    for (int i = 0; i < src_dim; ++i) {
      int dim_index = dst_index / dst_stride[i];
      dst_index = dst_index % dst_stride[i];
      src_index += dim_index * src_stride[permute[i]];
    }
    dst[index] = src[src_index];
  }
}

template <class scalar_t>
void memcpyPermute(scalar_t *dst, const scalar_t *src, int *src_size, int *permute, int src_dim,
                   cudaStream_t stream) {
  size_t copy_size = 1;
  TensorDesc ts_permute;
  memcpy(&(ts_permute.shape[0]), permute, src_dim * sizeof(int));

  TensorDesc ts_src_stride;
  TensorDesc ts_dst_stride;
  ts_src_stride.dim = src_dim;
  ts_dst_stride.dim = src_dim;
  int *src_stride = &(ts_src_stride.stride[0]);
  int *dst_stride = &(ts_dst_stride.stride[0]);
  int *dst_size = &(ts_dst_stride.shape[0]);
  src_stride[src_dim - 1] = 1;
  dst_stride[src_dim - 1] = 1;

  for (int i = src_dim - 1; i >= 0; --i) {
    dst_size[i] = src_size[permute[i]];
    if (i < src_dim - 1) {
      src_stride[i] = src_stride[i + 1] * src_size[i + 1];
    }
  }

  for (int i = src_dim - 1; i >= 0; --i) {
    copy_size *= dst_size[i];
    if (i < src_dim - 1) {
      dst_stride[i] = dst_stride[i + 1] * dst_size[i + 1];
    }
  }

  copy_permute_kernel<scalar_t><<<GET_BLOCKS(copy_size), THREADS_PER_BLOCK, 0, stream>>>(
      dst, src, copy_size, ts_src_stride, ts_dst_stride, ts_permute);
}

template void memcpyPermute<float>(float *dst, const float *src, int *src_size, int *permute,
                                   int src_dim, cudaStream_t stream);
template void memcpyPermute<__half>(__half *dst, const __half *src, int *src_size, int *permute,
                                  int src_dim, cudaStream_t stream);


template <>
cublasStatus_t cublasGemmWrap<float>(cublasHandle_t handle, cublasOperation_t transa,
                                     cublasOperation_t transb, int m, int n, int k,
                                     const float *alpha, const float *A, int lda, const float *B,
                                     int ldb, const float *beta, float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasGemmWrap<__half>(cublasHandle_t handle, cublasOperation_t transa,
                                    cublasOperation_t transb, int m, int n, int k,
                                    const __half *alpha, const __half *A, int lda, const __half *B,
                                    int ldb, const __half *beta, __half *C, int ldc) {
  return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
