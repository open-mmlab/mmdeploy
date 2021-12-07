// Copyright (c) OpenMMLab. All rights reserved.
#include "common_cuda_helper.hpp"
#include "float.h"
#include "trt_roi_align_kernel.hpp"

/*** Forward ***/
template <typename T>
__global__ void roi_align_forward_cuda_kernel(const int nthreads, const T* input, const T* rois,
                                              T* output, T* argmax_y, T* argmax_x,
                                              const int pooled_height, const int pooled_width,
                                              const T spatial_scale, const int sampling_ratio,
                                              const int pool_mode,  // 0 - max pool, 1 - avg pool
                                              const bool aligned, const int channels,
                                              const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {  // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input = input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_height / pooled_height));
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_width / pooled_width));

    if (pool_mode == 0) {
      // We do max pooling inside a bin
      T maxval = -FLT_MAX;
      T maxidx_y = -1.f, maxidx_x = -1.f;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
          T val = bilinear_interpolate(offset_input, height, width, y, x);
          if (val > maxval) {
            maxval = val;
            maxidx_y = y;
            maxidx_x = x;
          }
        }
      }
      output[index] = maxval;
      argmax_y[index] = maxidx_y;
      argmax_x[index] = maxidx_x;
    } else if (pool_mode == 1) {
      // We do average pooling inside a bin
      const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
      T output_val = 0.;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
          T val = bilinear_interpolate(offset_input, height, width, y, x);
          output_val += val;
        }
      }
      output[index] = output_val / count;
    }
  }
}

template <typename scalar_t>
void TRTRoIAlignForwardCUDAKernelLauncher(const scalar_t* input, const scalar_t* rois,
                                          scalar_t* output, scalar_t* argmax_y, scalar_t* argmax_x,
                                          int output_size, int channels, int height, int width,
                                          int aligned_height, int aligned_width,
                                          scalar_t spatial_scale, int sampling_ratio, int pool_mode,
                                          bool aligned, cudaStream_t stream) {
  roi_align_forward_cuda_kernel<scalar_t>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          output_size, input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
          static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode, aligned, channels,
          height, width);
}

template void TRTRoIAlignForwardCUDAKernelLauncher<float>(
    const float* input, const float* rois, float* output, float* argmax_y, float* argmax_x,
    int output_size, int channels, int height, int width, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, int pool_mode, bool aligned, cudaStream_t stream);
