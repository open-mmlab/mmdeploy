// Modified from
// https://github.com/pytorch/pytorch/blob/6adbe044e39c8e8db158d91e151aa6dead6e9aa4/aten/src/ATen/native/cuda/UpSampleBicubic2d.cu
#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "common_cuda_helper.hpp"
#include "trt_deform_conv_v3_kernel.hpp"

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

template <typename scalar_t>
__device__ scalar_t dcnv3_im2col_bilinear(const scalar_t *&bottom_data, const int &height,
                                          const int &width, const int &group,
                                          const int &group_channels, const scalar_t &h,
                                          const scalar_t &w, const int &g, const int &c) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = group * group_channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = g * group_channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }
  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__global__ void dcnv3_im2col_gpu_kernel(
    const int num_kernels, const scalar_t *data_im, const scalar_t *data_offset,
    const scalar_t *data_mask, scalar_t *data_col, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out, const scalar_t offset_scale) {
  CUDA_KERNEL_LOOP(index, num_kernels) {
    int _temp = index;
    const int c_col = _temp % group_channels;
    _temp /= group_channels;
    const int sampling_index = _temp;
    const int g_col = _temp % group;
    _temp /= group;
    const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_temp % width_out) * stride_w;
    _temp /= width_out;
    const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_temp % height_out) * stride_h;
    _temp /= height_out;
    const int b_col = _temp;

    const int input_size = height_in * width_in;
    scalar_t *data_col_ptr = data_col + index;
    const int kernel_size = kernel_h * kernel_w;
    int data_weight_ptr = sampling_index * kernel_size;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = group * group_channels;
    scalar_t col = 0;
    const scalar_t *data_im_ptr = data_im + b_col * input_size * qid_stride;
    // top-left
    const scalar_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
    const scalar_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
    for (int i = 0; i < kernel_w; ++i) {
      for (int j = 0; j < kernel_h; ++j) {
        const scalar_t offset_w = data_offset[data_loc_w_ptr];
        const scalar_t offset_h = data_offset[data_loc_w_ptr + 1];
        const scalar_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale;
        const scalar_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;
        const scalar_t weight = data_mask[data_weight_ptr];
        if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in) {
          col += dcnv3_im2col_bilinear(data_im_ptr, height_in, width_in, group, group_channels,
                                       loc_h, loc_w, g_col, c_col) *
                 weight;
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t>
void dcnv3_im2col_cuda(cudaStream_t stream, const scalar_t *data_im, const scalar_t *data_offset,
                       const scalar_t *data_mask, scalar_t *data_col, const int kernel_h,
                       const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
                       const int pad_w, const int dilation_h, const int dilation_w, const int group,
                       const int group_channels, const int batch_n, const int height_in,
                       const int width_in, const int height_out, const int width_out,
                       const scalar_t offset_scale) {
  const int num_kernels = batch_n * height_out * width_out * group * group_channels;
  const int num_actual_kernels = batch_n * height_out * width_out * group * group_channels;
  const int num_threads = CUDA_NUM_THREADS;
  dcnv3_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
          num_kernels, data_im, data_offset, data_mask, data_col, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, height_in,
          width_in, height_out, width_out, offset_scale);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in dcnv3_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void dcnv3_cuda_forward(const scalar_t *input, const scalar_t *offset, const scalar_t *mask,
                        scalar_t *output, int batch, int channels, int height_in, int width_in,
                        const int kernel_h, const int kernel_w, const int stride_h,
                        const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
                        const int dilation_w, const int group, const int group_channels,
                        const float offset_scale, const int im2col_step, cudaStream_t stream) {
  const int height_out = (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out = (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int im2col_step_ = std::min(batch, im2col_step);

  const int batch_n = im2col_step_;
  auto per_input_size = height_in * width_in * group * group_channels;
  auto per_output_size = height_out * width_out * group * group_channels;
  auto per_offset_size = height_out * width_out * group * kernel_h * kernel_w * 2;
  auto per_mask_size = height_out * width_out * group * kernel_h * kernel_w;
  for (int n = 0; n < batch / im2col_step_; ++n) {
    // AT_DISPATCH_FLOATING_TYPES(
    dcnv3_im2col_cuda<scalar_t>(
        stream, input + n * im2col_step_ * per_input_size,
        offset + n * im2col_step_ * per_offset_size, mask + n * im2col_step_ * per_mask_size,
        output + n * im2col_step_ * per_output_size, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group, group_channels, batch_n, height_in, width_in,
        height_out, width_out, offset_scale);
  }
}

template <typename scalar_t>
void DeformConvv3ForwardCUDAKernelLauncher(
    const scalar_t *input, const scalar_t *offset, const scalar_t *mask, scalar_t *output,
    void *workspace, int batch, int channels, int height, int width, int channels_out, int kernel_w,
    int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h, int dilation_w, int dilation_h,
    int group, int group_channel, float offset_scale, int im2col_step, cudaStream_t stream) {
  dcnv3_cuda_forward(input, offset, mask, output, batch, channels, height, width, kernel_h,
                     kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                     group_channel, offset_scale, im2col_step, stream);
}

template void DeformConvv3ForwardCUDAKernelLauncher<float>(
    const float *input, const float *offset, const float *mask, float *output, void *workspace,
    int batch, int channels, int height, int width, int channels_out, int kernel_w, int kernel_h,
    int stride_w, int stride_h, int pad_w, int pad_h, int dilation_w, int dilation_h, int group,
    int group_channel, float offset_scale, int im2col_step, cudaStream_t stream);
