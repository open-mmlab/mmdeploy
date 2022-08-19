// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_SCALED_DOT_PRODUCT_ATTENTION_KERNEL_HPP
#define TRT_SCALED_DOT_PRODUCT_ATTENTION_KERNEL_HPP
#include <cuda_runtime.h>

template <typename scalar_t>
void dot_product_attention_impl(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                                const scalar_t* mask, scalar_t* attn, scalar_t* weight, int B,
                                int Nt, int Ns, int E, int mask_dim, cudaStream_t stream);

#endif
