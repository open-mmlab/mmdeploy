// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_SCALED_DOT_PRODUCT_ATTENTION_KERNEL_HPP
#define TRT_SCALED_DOT_PRODUCT_ATTENTION_KERNEL_HPP
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

template <typename scalar_t>
void dot_product_attention_impl(const scalar_t* query, const scalar_t* key, const scalar_t* value,
                                const scalar_t* mask, scalar_t* attn, scalar_t* output, int B,
                                int Nt, int Ns, int E, const int* mask_dims,
                                cudnnTensorDescriptor_t& x_desc, cudnnTensorDescriptor_t& y_desc,
                                cudnnTensorDescriptor_t& mask_desc, cudnnDataType_t cudnn_dtype,
                                cudaStream_t stream, cublasHandle_t cublas_handle,
                                cudnnHandle_t cudnn_handle);

#endif
