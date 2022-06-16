// Copyright (c) OpenMMLab. All rights reserved.
#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "common_cuda_helper.hpp"
#include "trt_multi_level_roi_align_kernel.hpp"
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

template <typename scalar_t, bool aligned, int pool_mode>
__device__ scalar_t roi_align_single(const scalar_t *__restrict__ bottom_data,
                                     const int roi_batch_ind, const scalar_t roi_start_w,
                                     const scalar_t roi_start_h, const scalar_t roi_end_w,
                                     const scalar_t roi_end_h, const scalar_t spatial_scale,
                                     const int pw, const int ph, const int c, const int sample_num,
                                     const int channels, const int height, const int width,
                                     const int pooled_height, const int pooled_width) {
  // Force malformed ROIs to be 1x1
  scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)(aligned ? 0. : 1.));
  scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)(aligned ? 0. : 1.));

  const scalar_t bin_size_h = roi_height / pooled_height;
  const scalar_t bin_size_w = roi_width / pooled_width;

  const scalar_t *offset_bottom_data =
      bottom_data + (roi_batch_ind * channels + c) * height * width;

  const int sample_num_h = (sample_num > 0) ? sample_num : ceil(roi_height / pooled_height);
  const int sample_num_w = (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

  scalar_t output_val = (pool_mode == 0) ? -FLT_MAX : 0;
  const scalar_t y_offset = roi_start_h + ph * bin_size_h;
  const scalar_t y_scale = bin_size_h / (scalar_t)(sample_num_h);
  const scalar_t x_offset = roi_start_w + pw * bin_size_w;
  const scalar_t x_scale = bin_size_w / (scalar_t)(sample_num_w);
  for (int iy = 0; iy < sample_num_h; iy++) {
    const scalar_t y = fma(scalar_t(iy) + scalar_t(.5f), y_scale, y_offset);
    for (int ix = 0; ix < sample_num_w; ix++) {
      const scalar_t x = fma(scalar_t(ix) + scalar_t(.5f), x_scale, x_offset);
      scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data, height, width, y, x);
      if (pool_mode == 0) {
        output_val = max(output_val, val);
      } else {
        output_val += val;
      }
    }
  }
  if (pool_mode != 0) {
    output_val /= max(sample_num_h * sample_num_w, 1);
  }

  return output_val;
}

template <typename scalar_t, bool aligned>
__global__ void roi_extractor_kernel(scalar_t *__restrict__ output,
                                     const scalar_t *__restrict__ bottom_rois, FeatData feat_data,
                                     const int pool_mode, const int sample_num,
                                     const float roi_scale_factor, const int finest_scale,
                                     const int pooled_height, const int pooled_width,
                                     int nThreads) {
  CUDA_1D_KERNEL_LOOP(index, nThreads) {
    const int channels = feat_data.channels;
    int tmp_index = index;
    const int pw = tmp_index % pooled_width;
    tmp_index /= pooled_width;
    const int ph = tmp_index % pooled_height;
    tmp_index /= pooled_height;
    const int c = tmp_index % channels;
    const int n = tmp_index / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;

    scalar_t roi_offset_x0 = offset_bottom_rois[1];
    scalar_t roi_offset_y0 = offset_bottom_rois[2];
    scalar_t roi_offset_x1 = offset_bottom_rois[3];
    scalar_t roi_offset_y1 = offset_bottom_rois[4];

    const scalar_t scale = sqrtf((roi_offset_y1 - roi_offset_y0) * (roi_offset_x1 - roi_offset_x0));

    const int target_lvls =
        min(feat_data.num_featmap - 1,
            max(0, int(floorf(log2f(scale / (scalar_t)(finest_scale) + 1e-6)))));

    if (roi_scale_factor > 0.) {
      const scalar_t roi_off_cx = (roi_offset_x0 + roi_offset_x1) * 0.5;
      const scalar_t roi_off_cy = (roi_offset_y0 + roi_offset_y1) * 0.5;
      const scalar_t half_scale_factor = roi_scale_factor * 0.5;
      const scalar_t half_roi_off_w =
          fma(roi_offset_x1 - roi_offset_x0 + 1, half_scale_factor, scalar_t(-0.5));
      const scalar_t half_roi_off_h =
          fma(roi_offset_y1 - roi_offset_y0 + 1, half_scale_factor, scalar_t(-0.5));

      roi_offset_x0 = roi_off_cx - half_roi_off_w;
      roi_offset_x1 = roi_off_cx + half_roi_off_w;
      roi_offset_y0 = roi_off_cy - half_roi_off_h;
      roi_offset_y1 = roi_off_cy + half_roi_off_h;
    }

    const scalar_t spatial_scale = (scalar_t)feat_data.spatial_scale[target_lvls];
    const int height = feat_data.h[target_lvls];
    const int width = feat_data.w[target_lvls];
    const scalar_t *bottom_data = (scalar_t *)feat_data.data[target_lvls];

    const int roi_batch_ind = offset_bottom_rois[0];
    const scalar_t offset = aligned ? (scalar_t)-0.5 : (scalar_t)0.0;
    const scalar_t roi_start_w =
        fma(roi_offset_x0, spatial_scale, offset);  // roi_offset_x0 * spatial_scale + offset;
    const scalar_t roi_start_h =
        fma(roi_offset_y0, spatial_scale, offset);  // roi_offset_y0 * spatial_scale + offset;
    const scalar_t roi_end_w =
        fma(roi_offset_x1, spatial_scale, offset);  // (roi_offset_x1) * spatial_scale - offset;
    const scalar_t roi_end_h =
        fma(roi_offset_y1, spatial_scale, offset);  // (roi_offset_y1)*spatial_scale - offset;

    if (pool_mode == 0) {
      const scalar_t output_val = roi_align_single<scalar_t, aligned, 0>(
          bottom_data, roi_batch_ind, roi_start_w, roi_start_h, roi_end_w, roi_end_h, spatial_scale,
          pw, ph, c, sample_num, channels, height, width, pooled_height, pooled_width);
      output[index] = output_val;
    } else {
      const scalar_t output_val = roi_align_single<scalar_t, aligned, 1>(
          bottom_data, roi_batch_ind, roi_start_w, roi_start_h, roi_end_w, roi_end_h, spatial_scale,
          pw, ph, c, sample_num, channels, height, width, pooled_height, pooled_width);
      output[index] = output_val;
    }
  }
}

template <typename T>
void multi_level_roi_align(T *output, const T *rois, int num_rois, const void *const *feats,
                           int num_feats, int n, int c, int *h, int *w, float *strides,
                           int aligned_height, int aligned_width, int pool_mode, int sample_num,
                           float roi_scale_factor, int finest_scale, bool aligned,
                           cudaStream_t stream) {
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
    roi_extractor_kernel<T, true><<<GET_BLOCKS(nThreads), THREADS_PER_BLOCK, 0, stream>>>(
        output, rois, feat_data, pool_mode, sample_num, roi_scale_factor, finest_scale,
        aligned_height, aligned_width, nThreads);
  } else {
    roi_extractor_kernel<T, false><<<GET_BLOCKS(nThreads), THREADS_PER_BLOCK, 0, stream>>>(
        output, rois, feat_data, pool_mode, sample_num, roi_scale_factor, finest_scale,
        aligned_height, aligned_width, nThreads);
  }
}

template void multi_level_roi_align<float>(float *output, const float *rois, int num_rois,
                                           const void *const *feats, int num_feats, int n, int c,
                                           int *h, int *w, float *strides, int aligned_height,
                                           int aligned_width, int pool_mode, int sample_num,
                                           float roi_scale_factor, int finest_scale, bool aligned,
                                           cudaStream_t stream);
