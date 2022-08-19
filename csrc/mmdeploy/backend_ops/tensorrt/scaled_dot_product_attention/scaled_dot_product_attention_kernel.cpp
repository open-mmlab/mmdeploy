// Copyright (c) OpenMMLab. All rights reserved
#include "scaled_dot_product_attention_kernel.hpp"

#include "common_cuda_helper.hpp"
#include "trt_plugin_helper.hpp"

template <typename scalar_t>
__global__ void dot_product_attention_kernel(const scalar_t* q, const scalar_t* k,
                                             const scalar_t* v, const scalar_t* mask,
                                             scalar_t* attn, scalar_t* weight, int B, int Nt,
                                             int Ns, int E, int mask_dim) {
  // cache q line
  // loop over k(NS), dot product(wrap level?), cache q@k line
  //
}

template <typename scalar_t>
void dot_product_attention_impl(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                                const scalar_t* mask, scalar_t* attn, scalar_t* weight, int B,
                                int Nt, int Ns, int E, int mask_dim, cudaStream_t stream) {}

template void dot_product_attention_impl<float>(const float* q, const float* k, const float* v,
                                                const float* mask, float* attn, float* weight,
                                                int B, int Nt, int Ns, int E, int mask_dim,
                                                cudaStream_t stream);
