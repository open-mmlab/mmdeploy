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

#include "common_cuda_helper.hpp"
#include "trt_deform_conv_kernel.cuh"
#include "trt_deform_conv_kernel.hpp"
#include "trt_plugin_helper.hpp"

template <typename scalar_t>
void deform_conv_im2col(const scalar_t* input, const scalar_t* offset, scalar_t* column,
                        const int channels, const int height, const int width, const int ksize_h,
                        const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
                        const int stride_w, const int dilation_h, const int dilation_w,
                        const int parallel_imgs, const int deformable_group, cudaStream_t stream) {
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
      num_kernels, input, offset, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, channel_per_deformable_group, parallel_imgs, channels,
      deformable_group, height_col, width_col, column);

  cudaCheckError();
}

template <typename scalar_t>
void deform_conv(const scalar_t* input, const scalar_t* weight, const scalar_t* offset,
                 scalar_t* output, void* workspace, int batchSize, int nInputPlane, int inputHeight,
                 int inputWidth, int nOutputPlane, int kW, int kH, int dW, int dH, int padW,
                 int padH, int dilationW, int dilationH, int group, int deformable_group,
                 int im2col_step, cublasHandle_t cublas_handle, cudaStream_t stream) {
  size_t word_size = sizeof(scalar_t);

  im2col_step = std::min(int(batchSize), im2col_step);
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long outputHW = outputHeight * outputWidth;
  long kHW = kH * kW;
  long columns_size =
      mmdeploy::getAlignedSize(nInputPlane * kHW * im2col_step * outputHW * word_size);

  // column buffer for img2col
  char* workspace_ptr = reinterpret_cast<char*>(workspace);
  scalar_t* columns = reinterpret_cast<scalar_t*>(workspace_ptr);
  workspace_ptr = workspace_ptr + columns_size;

  scalar_t* output_buffer;
  if (im2col_step == 1) {
    output_buffer = output;
  } else {
    // output need permute when im2col_step!=1
    output_buffer = reinterpret_cast<scalar_t*>(workspace_ptr);
  }

  long input_elt_step = im2col_step * nInputPlane * inputHeight * inputWidth;
  long offset_elt_step = im2col_step * deformable_group * 2 * kHW * outputHW;
  long out_buffer_step = nOutputPlane * im2col_step * outputHW;
  long col_g_step = nInputPlane * kHW * im2col_step * outputHW / group;
  long weight_g_step = nOutputPlane * nInputPlane * kHW / (group * group);
  long out_buffer_g_step = out_buffer_step / group;
  int m = nOutputPlane / group;
  int n = im2col_step * outputHW;
  int k = nInputPlane * kHW / group;
  scalar_t alpha = 1.f;
  scalar_t beta = 0.f;

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    const scalar_t* input_start = input + elt * input_elt_step;
    const scalar_t* offset_start = offset + elt * offset_elt_step;

    deform_conv_im2col<scalar_t>(input_start, offset_start, columns, nInputPlane, inputHeight,
                                 inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
                                 im2col_step, deformable_group, stream);

    for (int g = 0; g < group; ++g) {
      const scalar_t* weight_start = weight + g * weight_g_step;
      scalar_t* col_start = columns + g * col_g_step;
      scalar_t* out_buffer_start = output_buffer + elt * out_buffer_step + g * out_buffer_g_step;

      cublasGemmWrap<scalar_t>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, col_start,
                               n, weight_start, k, &beta, out_buffer_start, n);
      cudaCheckError();
    }
  }

  if (im2col_step != 1) {
    int output_buffer_shape[5] = {batchSize / im2col_step, nOutputPlane, im2col_step,
                                  static_cast<int>(outputHeight), static_cast<int>(outputWidth)};
    int output_buffer_permute[5] = {0, 2, 1, 3, 4};
    memcpyPermute<scalar_t>(output, output_buffer, &output_buffer_shape[0],
                            &output_buffer_permute[0], 5, stream);
  }
}

template void deform_conv<float>(const float* input, const float* weight, const float* offset,
                                 float* output, void* workspace, int batchSize, int nInputPlane,
                                 int inputHeight, int inputWidth, int nOutputPlane, int kW, int kH,
                                 int dW, int dH, int padW, int padH, int dilationW, int dilationH,
                                 int group, int deformable_group, int im2col_step,
                                 cublasHandle_t cublas_handle, cudaStream_t stream);

template void deform_conv<__half>(const __half* input, const __half* weight, const __half* offset,
                                  __half* output, void* workspace, int batchSize, int nInputPlane,
                                  int inputHeight, int inputWidth, int nOutputPlane, int kW, int kH,
                                  int dW, int dH, int padW, int padH, int dilationW, int dilationH,
                                  int group, int deformable_group, int im2col_step,
                                  cublasHandle_t cublas_handle, cudaStream_t stream);
