/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#include <cuda_fp16.h>

#include "common_cuda_helper.hpp"

template <typename scalar_t>
__device__ __forceinline__ scalar_t deformable_im2col_bilinear(const scalar_t* __restrict__ input,
                                                               const int height, const int width,
                                                               float h, float w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  const int h_low = floorf(h);
  const int w_low = floorf(w);

  input += h_low * width;
  const scalar_t v1 = (h_low >= 0 && w_low >= 0) ? input[w_low] : static_cast<scalar_t>(0.0f);
  const int w_high = w_low + 1;
  const scalar_t v2 =
      (h_low >= 0 && w_high <= width - 1) ? input[w_high] : static_cast<scalar_t>(0.0f);
  const scalar_t lw = w - w_low;
  const scalar_t v_low = fmaf(v2 - v1, lw, v1);
  input += width;
  const scalar_t v3 =
      (h_low <= height - 2 && w_low >= 0) ? input[w_low] : static_cast<scalar_t>(0.0f);
  const scalar_t v4 =
      (h_low <= height - 2 && w_high <= width - 1) ? input[w_high] : static_cast<scalar_t>(0.0f);
  const scalar_t v_high = fmaf(v4 - v3, lw, v3);
  const scalar_t lh = h - h_low;
  const scalar_t val = fmaf(v_high - v_low, lh, v_low);
  return val;
}

template <>
__device__ __forceinline__ __half deformable_im2col_bilinear(const __half* __restrict__ input,
                                                             const int height, const int width,
                                                             float h, float w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  const int h_low = floorf(h);
  const int w_low = floorf(w);

  input += h_low * width;
  const float v1 = (h_low >= 0 && w_low >= 0) ? __half2float(input[w_low]) : 0.0f;
  const int w_high = w_low + 1;
  const float v2 = (h_low >= 0 && w_high <= width - 1) ? __half2float(input[w_high]) : 0.0f;
  const float lw = w - w_low;
  const float v_low = fmaf(v2 - v1, lw, v1);
  input += width;
  const float v3 = (h_low <= height - 2 && w_low >= 0) ? __half2float(input[w_low]) : 0.0f;
  const float v4 =
      (h_low <= height - 2 && w_high <= width - 1) ? __half2float(input[w_high]) : 0.0f;
  const float v_high = fmaf(v4 - v3, lw, v3);
  const float lh = h - h_low;
  const float val = fmaf(v_high - v_low, lh, v_low);
  return __float2half(val);
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
    const int n, const scalar_t* __restrict__ data_im, const scalar_t* __restrict__ data_offset,
    const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col, const int width_col,
    scalar_t* __restrict__ data_col) {
  const int hw_col = height_col * width_col;
  const int data_col_step = batch_size * hw_col;

  CUDA_1D_KERNEL_LOOP(index, n) {
    // index index of output matrix
    int tmp_index = index;
    const int w_col = tmp_index % width_col;
    tmp_index /= width_col;
    const int h_col = tmp_index % height_col;
    tmp_index /= height_col;
    const int b_col = tmp_index % batch_size;
    const int c_im = tmp_index / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    scalar_t* __restrict__ data_col_ptr = data_col + c_col * data_col_step + index % data_col_step;
    const scalar_t* __restrict__ data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t* __restrict__ data_offset_ptr =
        data_offset +
        ((b_col * deformable_group + deformable_group_index) << 1) * kernel_h * kernel_w * hw_col +
        h_col * width_col + w_col;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h = (i * kernel_w + j) * hw_col << 1;
        const scalar_t offset_h = data_offset_ptr[data_offset_h];
        const int data_offset_w = data_offset_h + hw_col;
        const scalar_t offset_w = data_offset_ptr[data_offset_w];
        const scalar_t h_im = h_in + i * dilation_h + (float)offset_h;
        const scalar_t w_im = w_in + j * dilation_w + (float)offset_w;
        const scalar_t val = deformable_im2col_bilinear(data_im_ptr, height, width, h_im, w_im);
        *data_col_ptr = val;
        data_col_ptr += data_col_step;
      }
    }
  }
}
