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
void dot_product_attention_impl(const scalar_t* query, const scalar_t* key, const scalar_t* value,
                                const scalar_t* mask, scalar_t* attn, scalar_t* output, int B,
                                int Nt, int Ns, int E, const int* mask_dims,
                                cudnnTensorDescriptor_t& x_desc, cudnnTensorDescriptor_t& y_desc,
                                cudnnTensorDescriptor_t& mask_desc, cudnnDataType_t cudnn_dtype,
                                cudaStream_t stream, cublasHandle_t cublas_handle,
                                cudnnHandle_t cudnn_handle) {
  {
    // Q @ K
    const int m = Ns;
    const int n = Nt;
    const int k = E;
    const auto alpha = scalar_t(1.0f / sqrt(float(E)));
    const auto beta = scalar_t(0);
    cublasgemmStridedBatchedWrap(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, key, k,
                                 Ns * E, query, k, Nt * E, &beta, attn, m, Nt * Ns, B);
  }

  if (mask_dims != nullptr && mask_dims[0] != 0) {
    const auto alpha = scalar_t(1);
    const auto beta = scalar_t(1);
    cudnnSetTensor4dDescriptor(mask_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, mask_dims[0],
                               mask_dims[1], mask_dims[2]);
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, B, Nt, Ns);
    cudnnAddTensor(cudnn_handle, &alpha, mask_desc, mask, &beta, x_desc, attn);
  }

  {
    // softmax attention
    const auto alpha = scalar_t(1);
    const auto beta = scalar_t(0);
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
    const auto alpha = scalar_t(1);
    const auto beta = scalar_t(0);
    cublasgemmStridedBatchedWrap(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, value, m,
                                 Ns * E, (const scalar_t*)(attn), k, Ns * Nt, &beta, output, m,
                                 Nt * E, B);
  }
}

template void dot_product_attention_impl<float>(
    const float* query, const float* key, const float* value, const float* mask, float* attn,
    float* output, int B, int Nt, int Ns, int E, const int* mask_dims,
    cudnnTensorDescriptor_t& x_desc, cudnnTensorDescriptor_t& y_desc,
    cudnnTensorDescriptor_t& mask_desc, cudnnDataType_t cudnn_dtype, cudaStream_t stream,
    cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle);
