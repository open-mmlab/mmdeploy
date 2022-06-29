// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_GRID_PRIORS_KERNEL_HPP
#define TRT_GRID_PRIORS_KERNEL_HPP
#include <cuda_runtime.h>

template <typename scalar_t>
void trt_grid_priors_impl(const scalar_t* base_anchor, scalar_t* output, int num_base_anchors,
                          int feat_w, int feat_h, int stride_w, int stride_h, cudaStream_t stream);

#endif
