// Copyright (c) OpenMMLab. All rights reserved.
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

template <typename scalar_t>
__device__ scalar_t roi_align_single(const scalar_t *bottom_data, const int roi_batch_ind,
                                     const scalar_t roi_start_w, const scalar_t roi_start_h,
                                     const scalar_t roi_end_w, const scalar_t roi_end_h,
                                     const scalar_t spatial_scale, const int pw, const int ph,
                                     const int c, const int sample_num, const int channels,
                                     const int height, const int width, const int pooled_height,
                                     const int pooled_width, const bool aligned) {
  // Force malformed ROIs to be 1x1
  scalar_t roi_width = fmaxf((scalar_t)roi_end_w - (scalar_t)roi_start_w, 0.);
  scalar_t roi_height = fmaxf((scalar_t)roi_end_h - (scalar_t)roi_start_h, 0.);
  if (!aligned) {
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
  }

  const scalar_t bin_size_h = roi_height / pooled_height;
  const scalar_t bin_size_w = roi_width / pooled_width;

  const scalar_t *offset_bottom_data =
      bottom_data + (roi_batch_ind * channels + c) * height * width;

  int sample_num_h = (sample_num > 0) ? sample_num : ceil(roi_height / pooled_height);  // e.g., = 2
  int sample_num_w = (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

  scalar_t output_val = 0;
#pragma unroll
  for (int iy = 0; iy < sample_num_h; iy++) {
    const scalar_t y = roi_start_h + ph * bin_size_h +
                       (scalar_t)(iy + scalar_t(.5f)) * bin_size_h / (scalar_t)(sample_num_h);
#pragma unroll
    for (int ix = 0; ix < sample_num_w; ix++) {
      const scalar_t x = roi_start_w + pw * bin_size_w +
                         (scalar_t)(ix + scalar_t(.5f)) * bin_size_w / (scalar_t)(sample_num_w);
      scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data, height, width, y, x);
      output_val += val;
    }
  }
  output_val /= max(sample_num_h * sample_num_w, 1);

  return output_val;
}

template <typename scalar_t>
__global__ void roi_extractor_kernel(scalar_t *output, const scalar_t *bottom_rois,
                                     FeatData feat_data, const int sample_num,
                                     const float roi_scale_factor, const int finest_scale,
                                     const int pooled_height, const int pooled_width,
                                     const bool aligned, int nThreads) {
  CUDA_1D_KERNEL_LOOP(index, nThreads) {
    const int channels = feat_data.channels;
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;

    scalar_t roi_offset_x0 = offset_bottom_rois[1];
    scalar_t roi_offset_y0 = offset_bottom_rois[2];
    scalar_t roi_offset_x1 = offset_bottom_rois[3];
    scalar_t roi_offset_y1 = offset_bottom_rois[4];

    const scalar_t scale = sqrtf((roi_offset_y1 - roi_offset_y0) * (roi_offset_x1 - roi_offset_x0));

    const int target_lvls = fminf(feat_data.num_featmap - 1,
                                  fmaxf(0, floorf(log2f(scale / (scalar_t)(finest_scale) + 1e-6))));

    if (roi_scale_factor > 0.) {
      const scalar_t roi_off_cx = (roi_offset_x0 + roi_offset_x1) * 0.5;
      const scalar_t roi_off_cy = (roi_offset_y0 + roi_offset_y1) * 0.5;
      const scalar_t roi_off_w = (roi_offset_x1 - roi_offset_x0 + 1) * roi_scale_factor;
      const scalar_t roi_off_h = (roi_offset_y1 - roi_offset_y0 + 1) * roi_scale_factor;

      roi_offset_x0 = roi_off_cx - roi_off_w * 0.5 + 0.5;
      roi_offset_x1 = roi_off_cx + roi_off_w * 0.5 - 0.5;
      roi_offset_y0 = roi_off_cy - roi_off_h * 0.5 + 0.5;
      roi_offset_y1 = roi_off_cy + roi_off_h * 0.5 - 0.5;
    }

    const scalar_t spatial_scale = (scalar_t)feat_data.spatial_scale[target_lvls];
    const int height = feat_data.h[target_lvls];
    const int width = feat_data.w[target_lvls];
    const scalar_t *bottom_data = (scalar_t *)feat_data.data[target_lvls];

    const int roi_batch_ind = offset_bottom_rois[0];
    const scalar_t offset = aligned ? (scalar_t)0.5 : (scalar_t)0.0;
    const scalar_t roi_start_w = roi_offset_x0 * spatial_scale - offset;
    const scalar_t roi_start_h = roi_offset_y0 * spatial_scale - offset;
    const scalar_t roi_end_w = (roi_offset_x1)*spatial_scale - offset;
    const scalar_t roi_end_h = (roi_offset_y1)*spatial_scale - offset;

    const scalar_t output_val = roi_align_single<scalar_t>(
        bottom_data, roi_batch_ind, roi_start_w, roi_start_h, roi_end_w, roi_end_h, spatial_scale,
        pw, ph, c, sample_num, channels, height, width, pooled_height, pooled_width, aligned);

    output[index] = output_val;
  }
}

template <typename T>
void multi_level_roi_align(T *output, const T *rois, int num_rois, const void *const *feats,
                           int num_feats, int n, int c, int *h, int *w, float *strides,
                           int aligned_height, int aligned_width, int sample_num,
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
  // bool aligned = true;
  roi_extractor_kernel<T><<<GET_BLOCKS(nThreads), THREADS_PER_BLOCK, 0, stream>>>(
      output, rois, feat_data, sample_num, roi_scale_factor, finest_scale, aligned_height,
      aligned_width, aligned, nThreads);
}

template void multi_level_roi_align<float>(float *output, const float *rois, int num_rois,
                                           const void *const *feats, int num_feats, int n, int c,
                                           int *h, int *w, float *strides, int aligned_height,
                                           int aligned_width, int sample_num,
                                           float roi_scale_factor, int finest_scale, bool aligned,
                                           cudaStream_t stream);
