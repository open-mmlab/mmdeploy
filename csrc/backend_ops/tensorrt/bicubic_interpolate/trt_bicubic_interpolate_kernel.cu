// Modified from
// https://github.com/pytorch/pytorch/blob/6adbe044e39c8e8db158d91e151aa6dead6e9aa4/aten/src/ATen/native/cuda/UpSampleBicubic2d.cu
#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "common_cuda_helper.hpp"
#include "trt_bicubic_interpolate_kernel.hpp"

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename scalar_t>
__device__ __forceinline__ static scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename scalar_t>
__device__ __forceinline__ static scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename scalar_t>
__device__ __forceinline__ static void get_cubic_upsample_coefficients(scalar_t coeffs[4],
                                                                       scalar_t t) {
  scalar_t A = -0.75;

  scalar_t x1 = t;
  coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<scalar_t>(x1, A);

  // opposite coefficients
  scalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);
}

template <typename scalar_t>
__device__ __forceinline__ static scalar_t cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2,
                                                          scalar_t x3, scalar_t t) {
  scalar_t coeffs[4];
  get_cubic_upsample_coefficients<scalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

/* Used by UpSampleBicubic2d.cu */
template <typename scalar_t>
__device__ __forceinline__ static scalar_t upsample_get_value_bounded(const scalar_t *data,
                                                                      int batch, int channel,
                                                                      int batchsize, int channels,
                                                                      int height, int width, int y,
                                                                      int x) {
  int access_y = max(min(y, height - 1), 0);
  int access_x = max(min(x, width - 1), 0);
  return data[batch * channels * height * width + channel * height * width + access_y * width +
              access_x];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
area_pixel_compute_source_index(scalar_t scale, int64_t dst_index, bool align_corners, bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    scalar_t src_idx = scale * (dst_index + 0.5) - 0.5;
    // [Note] Follow Opencv resize logic:
    // We allow negative src_idx here and later will use
    //   dx = src_idx - floorf(src_idx)
    // to compute the "distance"(which affects weights).
    // For linear modes, weight distribution doesn't matter
    // for negative indices as they use 2 pixels to interpolate.
    // For example, [-1, 0], they both use pixel 0 value so it
    // doesn't affect if we bound the src_idx to 0 or not.
    // TODO: Our current linear mode impls use unbound indices
    // where we should and then remove this cubic flag.
    // This matters in cubic mode, as we might need [-1, 0, 1, 2]
    // to interpolate and the weights can be affected.
    return (!cubic && src_idx < 0) ? scalar_t(0) : src_idx;
  }
}

// cubic interpolation pytorch
template <typename scalar_t>
__global__ void resize_cubic_kernel_torch(const int num_elements, const scalar_t *src,
                                          const int batchsize, const int channels, int srcWidth,
                                          int srcHeight, scalar_t *dst, int dstWidth, int dstHeight,
                                          bool align_corners, float height_scale,
                                          float width_scale) {
  CUDA_1D_KERNEL_LOOP(index, num_elements) {
    // Special case: input and output are the same size, just copy
    const int output_x = index % dstWidth;
    const int output_y = index / dstWidth;

    if (srcHeight == dstHeight && srcWidth == dstWidth) {
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; c++) {
          const scalar_t val = src[n * channels * dstHeight * dstWidth + c * dstHeight * dstWidth +
                                   output_y * dstWidth + output_x];
          dst[n * channels * dstHeight * dstWidth + c * dstHeight * dstWidth + output_y * dstWidth +
              output_x] = val;
        }
      }
      return;
    }
    // Interpolation kernel
    scalar_t real_x =
        area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
    int in_x = floorf(real_x);
    scalar_t t_x = real_x - in_x;

    scalar_t real_y =
        area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
    int in_y = floorf(real_y);
    scalar_t t_y = real_y - in_y;

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; c++) {
        scalar_t coefficients[4];

        for (int k = 0; k < 4; k++) {
          coefficients[k] = cubic_interp1d<scalar_t>(
              upsample_get_value_bounded(src, n, c, batchsize, channels, srcHeight, srcWidth,
                                         in_y - 1 + k, in_x - 1),
              upsample_get_value_bounded(src, n, c, batchsize, channels, srcHeight, srcWidth,
                                         in_y - 1 + k, in_x + 0),
              upsample_get_value_bounded(src, n, c, batchsize, channels, srcHeight, srcWidth,
                                         in_y - 1 + k, in_x + 1),
              upsample_get_value_bounded(src, n, c, batchsize, channels, srcHeight, srcWidth,
                                         in_y - 1 + k, in_x + 2),
              t_x);
        }

        dst[n * channels * dstHeight * dstWidth + c * dstHeight * dstWidth + output_y * dstWidth +
            output_x] = scalar_t(cubic_interp1d(coefficients[0], coefficients[1], coefficients[2],
                                                coefficients[3], t_y));
      }
    }
  }
}

template <typename scalar_t>
void resizeGPU(const scalar_t *pIn_d, scalar_t *pOut_d, int batch, int channels, int srcWidth,
               int srcHeight, int dstWidth, int dstHeight, bool align_corners,
               cudaStream_t stream) {
  float height_scale = float(srcHeight) / dstHeight;
  float width_scale = float(srcWidth) / dstWidth;
  if (align_corners && dstWidth > 1 && dstHeight > 1) {
    height_scale = (float)(srcHeight - 1) / (dstHeight - 1);
    width_scale = (float)(srcWidth - 1) / (dstWidth - 1);
  }
  int n = batch * dstWidth * dstHeight * channels;
  resize_cubic_kernel_torch<<<GET_BLOCKS(n), THREADS_PER_BLOCK, 0, stream>>>(
      dstWidth * dstHeight, pIn_d, batch, channels, srcWidth, srcHeight, pOut_d, dstWidth,
      dstHeight, align_corners, height_scale, width_scale);
}

template <typename scalar_t>
void bicubic_interpolate(const scalar_t *input, scalar_t *output, int batch, int channels,
                         int in_height, int in_width, int out_height, int out_width,
                         bool align_corners, cudaStream_t stream) {
  resizeGPU(input, output, batch, channels, in_width, in_height, out_width, out_height,
            align_corners, stream);
}

template void bicubic_interpolate<float>(const float *input, float *output, int batch, int channels,
                                         int in_height, int in_width, int out_height, int out_width,
                                         bool align_corners, cudaStream_t stream);
