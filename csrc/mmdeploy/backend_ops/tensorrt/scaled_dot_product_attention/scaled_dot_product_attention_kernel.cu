// Copyright (c) OpenMMLab. All rights reserved
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <cmath>
#include <vector>

#include "common_cuda_helper.hpp"
#include "scaled_dot_product_attention_kernel.hpp"
#include "trt_plugin_helper.hpp"

template <typename scalar_t>
cublasStatus_t cublasgemmStridedBatchedWrap(cublasHandle_t handle, cublasOperation_t transa,
                                            cublasOperation_t transb, int m, int n, int k,
                                            const scalar_t* alpha, const scalar_t* A, int lda,
                                            long long int strideA, const scalar_t* B, int ldb,
                                            long long int strideB, const scalar_t* beta,
                                            scalar_t* C, int ldc, long long int strideC,
                                            int batchCount);

template <>
cublasStatus_t cublasgemmStridedBatchedWrap<float>(cublasHandle_t handle, cublasOperation_t transa,
                                                   cublasOperation_t transb, int m, int n, int k,
                                                   const float* alpha, const float* A, int lda,
                                                   long long int strideA, const float* B, int ldb,
                                                   long long int strideB, const float* beta,
                                                   float* C, int ldc, long long int strideC,
                                                   int batchCount) {
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
                                   strideB, beta, C, ldc, strideC, batchCount);
}

template <>
cublasStatus_t cublasgemmStridedBatchedWrap<__half>(cublasHandle_t handle, cublasOperation_t transa,
                                                    cublasOperation_t transb, int m, int n, int k,
                                                    const __half* alpha, const __half* A, int lda,
                                                    long long int strideA, const __half* B, int ldb,
                                                    long long int strideB, const __half* beta,
                                                    __half* C, int ldc, long long int strideC,
                                                    int batchCount) {
  return cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
                                   strideB, beta, C, ldc, strideC, batchCount);
}

template <typename scalar_t>
struct get_mask : public thrust::unary_function<int, scalar_t> {
  const scalar_t* _mask;
  int32_t _size;

  get_mask(const scalar_t* mask, int32_t size) : _mask(mask), _size(size) {}

  __host__ __device__ scalar_t operator()(int x) const { return _mask[x % _size]; }
};

template <typename scalar_t>
void dot_product_attention_impl(const scalar_t* query, const scalar_t* key, const scalar_t* value,
                                const scalar_t* mask, scalar_t* attn, scalar_t* weight, int B,
                                int Nt, int Ns, int E, int mask_dim,
                                cudnnTensorDescriptor_t& x_desc, cudnnTensorDescriptor_t& y_desc,
                                cudnnDataType_t cudnn_dtype, cudaStream_t stream,
                                cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle) {
  {
    // Q @ K
    const int m = Ns;
    const int n = Nt;
    const int k = E;
    auto alpha = scalar_t(1.0f / sqrt(float(E)));
    auto beta = scalar_t(0);
    cublasgemmStridedBatchedWrap(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, key, k,
                                 Ns * E, query, k, Nt * E, &beta, attn, m, Nt * Ns, B);
  }

  if (mask_dim != 0 && mask != nullptr) {
    thrust::plus<scalar_t> op;
    if (mask_dim == 3) {
      transform(thrust::cuda::par.on(stream), attn, attn + B * Nt * Ns, mask, attn, op);
    } else if (mask_dim == 2) {
      auto counting_iter = thrust::make_counting_iterator(0);
      auto trans_iter =
          thrust::make_transform_iterator(counting_iter, get_mask<scalar_t>(mask, Nt * Ns));
      transform(thrust::cuda::par.on(stream), attn, attn + B * Nt * Ns, trans_iter, attn, op);
    }
  }

  {
    // softmax attention
    auto alpha = scalar_t(1);
    auto beta = scalar_t(0);
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, B * Nt, Ns, 1, 1);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, B * Nt, Ns, 1, 1);
    cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
                        x_desc, attn, &beta, y_desc, attn);
  }

  {
    // attn @ v
    const int m = E;
    const int n = Nt;
    const int k = Ns;
    auto alpha = scalar_t(1);
    auto beta = scalar_t(0);
    cublasgemmStridedBatchedWrap(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, value, m,
                                 Ns * E, (const scalar_t*)(attn), k, Ns * Nt, &beta, weight, m,
                                 Nt * E, B);
  }
}

template void dot_product_attention_impl<float>(
    const float* query, const float* key, const float* value, const float* mask, float* attn,
    float* weight, int B, int Nt, int Ns, int E, int mask_dim, cudnnTensorDescriptor_t& x_desc,
    cudnnTensorDescriptor_t& y_desc, cudnnDataType_t cudnn_dtype, cudaStream_t stream,
    cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle);
