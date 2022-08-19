// Copyright (c) OpenMMLab. All rights reserved.
#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "common_cuda_helper.hpp"
#include "trt_multi_level_rotated_roi_align_kernel.hpp"
#include "trt_plugin_helper.hpp"

const int kMAX_FEATMAP_SIZE = 10;
struct FeatData {
  const void *data[kMAX_FEATMAP_SIZE];
  int batch_size;
  int channels;
  int h[kMAX_FEATMAP_SIZE];
  int w[kMAX_FEATMAP_SIZE];
  float spatial_scale[kMAX_FEATMAP_SIZE];
  int num_featmap;
};

template <typename scalar_t, bool aligned>
__device__ scalar_t roi_align_single(const scalar_t *__restrict__ bottom_data,
                                     const int roi_batch_ind, scalar_t roi_center_w,
                                     scalar_t roi_center_h, scalar_t roi_width, scalar_t roi_height,
                                     scalar_t theta, const scalar_t spatial_scale, const int pw,
                                     const int ph, const int c, const int sample_num,
                                     const int channels, const int height, const int width,
                                     const int pooled_height, const int pooled_width) {
  // Force malformed ROIs to be 1x1

  roi_width = max(roi_width, (scalar_t)1.);
  roi_height = max(roi_height, (scalar_t)1.);

  const scalar_t bin_size_h = roi_height / scalar_t(pooled_height);
  const scalar_t bin_size_w = roi_width / scalar_t(pooled_width);

  const scalar_t *offset_bottom_data =
      bottom_data + (roi_batch_ind * channels + c) * height * width;

  const int roi_bin_grid_h = (sample_num > 0) ? sample_num : ceil(roi_height / pooled_height);
  const int roi_bin_grid_w = (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

  const scalar_t roi_start_h = -roi_height / scalar_t(2.0);
  const scalar_t roi_start_w = -roi_width / scalar_t(2.0);
  const scalar_t cosscalar_theta = cos(theta);
  const scalar_t sinscalar_theta = sin(theta);

  // We do average (integral) pooling inside a bin
  const scalar_t count = max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

  scalar_t output_val = 0.;

  for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
    const scalar_t yy = roi_start_h + ph * bin_size_h +
                        static_cast<scalar_t>(iy + .5f) * bin_size_h /
                            static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
      const scalar_t xx =
          roi_start_w + pw * bin_size_w +
          static_cast<scalar_t>(ix + .5f) * bin_size_w / static_cast<scalar_t>(roi_bin_grid_w);

      // Rotate by theta (counterclockwise) around the center and translate
      scalar_t y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;
      scalar_t x = yy * sinscalar_theta + xx * cosscalar_theta + roi_center_w;

      scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data, height, width, y, x);
      output_val += val;
    }
  }

  return output_val / count;
}

template <typename scalar_t, bool aligned>
__global__ void rotated_roi_extractor_kernel(scalar_t *__restrict__ output,
                                             const scalar_t *__restrict__ bottom_rois,
                                             FeatData feat_data, const int clockwise,
                                             const int sample_num, const float roi_scale_factor,
                                             const int finest_scale, const int pooled_height,
                                             const int pooled_width, int nThreads) {
  CUDA_1D_KERNEL_LOOP(index, nThreads) {
    const int channels = feat_data.channels;
    int tmp_index = index;
    const int pw = tmp_index % pooled_width;
    tmp_index /= pooled_width;
    const int ph = tmp_index % pooled_height;
    tmp_index /= pooled_height;
    const int c = tmp_index % channels;
    const int n = tmp_index / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 6;

    scalar_t roi_offset_x0 = offset_bottom_rois[1];
    scalar_t roi_offset_y0 = offset_bottom_rois[2];
    scalar_t roi_offset_width = offset_bottom_rois[3];
    scalar_t roi_offset_height = offset_bottom_rois[4];
    scalar_t theta = offset_bottom_rois[5];

    const scalar_t scale = sqrtf(roi_offset_width * roi_offset_height);

    const int target_lvls =
        min(feat_data.num_featmap - 1,
            max(0, int(floorf(log2f(scale / (scalar_t)(finest_scale) + 1e-6)))));

    if (roi_scale_factor > 0.) {
      roi_offset_width = roi_offset_width * roi_scale_factor;
      roi_offset_height = roi_offset_height * roi_scale_factor;
    }

    const scalar_t spatial_scale = (scalar_t)feat_data.spatial_scale[target_lvls];
    const int height = feat_data.h[target_lvls];
    const int width = feat_data.w[target_lvls];
    const scalar_t *bottom_data = (scalar_t *)feat_data.data[target_lvls];

    const int roi_batch_ind = offset_bottom_rois[0];
    const scalar_t offset = aligned ? (scalar_t)-0.5 : (scalar_t)0.0;
    const scalar_t roi_center_w = fma(roi_offset_x0, spatial_scale, offset);
    const scalar_t roi_center_h = fma(roi_offset_y0, spatial_scale, offset);
    const scalar_t roi_width = roi_offset_width * spatial_scale;
    const scalar_t roi_height = roi_offset_height * spatial_scale;

    theta = clockwise > 0 ? -theta : theta;

    const scalar_t output_val = roi_align_single<scalar_t, aligned>(
        bottom_data, roi_batch_ind, roi_center_w, roi_center_h, roi_width, roi_height, theta,
        spatial_scale, pw, ph, c, sample_num, channels, height, width, pooled_height, pooled_width);
    output[index] = output_val;
  }
}

template <typename T>
void multi_level_rotated_roi_align(T *output, const T *rois, int num_rois, const void *const *feats,
                                   int num_feats, int n, int c, int *h, int *w, float *strides,
                                   int aligned_height, int aligned_width, int clockwise,
                                   int sample_num, float roi_scale_factor, int finest_scale,
                                   bool aligned, cudaStream_t stream) {
  FeatData feat_data;
  feat_data.batch_size = n;
  feat_data.channels = c;
  feat_data.num_featmap = num_feats;
  for (int i = 0; i < num_feats; ++i) {
    feat_data.data[i] = feats[i];
    feat_data.h[i] = h[i];
    feat_data.w[i] = w[i];
    feat_data.spatial_scale[i] = 1. / float(strides[i]);
  }
  int nThreads = num_rois * c * aligned_height * aligned_width;
  if (aligned) {
    rotated_roi_extractor_kernel<T, true><<<GET_BLOCKS(nThreads), THREADS_PER_BLOCK, 0, stream>>>(
        output, rois, feat_data, clockwise, sample_num, roi_scale_factor, finest_scale,
        aligned_height, aligned_width, nThreads);
  } else {
    rotated_roi_extractor_kernel<T, false><<<GET_BLOCKS(nThreads), THREADS_PER_BLOCK, 0, stream>>>(
        output, rois, feat_data, clockwise, sample_num, roi_scale_factor, finest_scale,
        aligned_height, aligned_width, nThreads);
  }
}

template void multi_level_rotated_roi_align<float>(
    float *output, const float *rois, int num_rois, const void *const *feats, int num_feats, int n,
    int c, int *h, int *w, float *strides, int aligned_height, int aligned_width, int clockwise,
    int sample_num, float roi_scale_factor, int finest_scale, bool aligned, cudaStream_t stream);
