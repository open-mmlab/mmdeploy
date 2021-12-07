// Copyright (c) OpenMMLab. All rights reserved.
#include "tensorslice.h"

#include <math.h>

#include "../ncnn_ops_definer.h"

namespace mmdeploy {
using namespace ncnn;
DEFINE_LAYER_CREATOR(TensorSlice)
DEFINE_NCNN_OPS(TensorSlice, TensorSlice)
TensorSlice::TensorSlice() {
  one_blob_only = true;
  support_inplace = false;
}

int TensorSlice::load_param(const ParamDict& pd) {
  starts = pd.get(0, Mat());
  ends = pd.get(1, Mat());
  axes = pd.get(2, Mat());
  steps = pd.get(3, Mat());
  if (axes.w == 0) {
    axes.create(starts.w);
    int* axes_ptr = axes;
    for (int i = 0; i < starts.w; i++) {
      axes_ptr[i] = i;
    }
  }
  if (steps.w == 0) {
    steps.create(axes.w);
    steps.fill(1);
  }
  return 0;
}

static inline int get_shape_by_axes(const Mat& blob, int axes, int dims) {
  switch (dims - axes) {
    case 0:
      return blob.w;
    case 1:
      return blob.h;
    case 2:
      return blob.c;
    default:
      fprintf(stderr, "wrong axes %d!\n", axes);
      return -1;
  }
  return 0;
}

int TensorSlice::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
  int dims = bottom_blob.dims;
  size_t elemsize = bottom_blob.elemsize;
  const int* start_ptr = starts;
  const int* end_ptr = ends;
  const int* axes_ptr = axes;
  const int* step_ptr = steps;
  if (starts.w > dims || ends.w > dims) {
    fprintf(stderr, "start/end attributes shape error!\n");
    return -100;
  }
  if (axes.w != 1) {
    fprintf(stderr,
            "axes.w must be 1 because any of multiaxes slice is regarded as "
            "multi-staged onnx slice in pytorch2onnx.");
  }
  if (dims == 1) {
    for (int i = 0; i < axes.w; i++) {
      int positive_axis = axes_ptr[i] < 0 ? dims + axes_ptr[i] : axes_ptr[i];
      int step = step_ptr[i];
      std::vector<float> temp_val;
      int start = start_ptr[i];
      int end = end_ptr[i];
      int cur = start;
      if (step > 0) {
        while (cur < end && cur < bottom_blob.w) {
          temp_val.push_back(bottom_blob[cur]);
          cur += step;
        }
      } else if (step < 0) {
        while (cur > end && cur > 0) {
          temp_val.push_back(bottom_blob[cur]);
          cur += step;
        }
      } else {
        fprintf(stderr, "step should not be 0!\n");
        return -100;
      }
      top_blob.create(temp_val.size(), elemsize, opt.blob_allocator);
      for (int i = 0; i < temp_val.size(); i++) {
        top_blob[i] = temp_val[i];
      }
    }
    return 0;
  }
  if (dims == 2) {
    std::vector<std::vector<int> > active_indice;
    std::vector<int> indices;
    for (int i = 0; i < bottom_blob.h; i++) {
      indices.push_back(i);
    }
    active_indice.push_back(indices);
    indices.clear();
    for (int i = 0; i < bottom_blob.w; i++) {
      indices.push_back(i);
    }
    active_indice.push_back(indices);
    for (int i = 0; i < axes.w; i++) {
      int positive_axis = axes_ptr[i] < 0 ? dims + axes_ptr[i] : axes_ptr[i];
      int step = step_ptr[i];
      int start = start_ptr[i];
      int end = end_ptr[i];
      int dim_shape = get_shape_by_axes(bottom_blob, positive_axis, dims);
      int dim_shape_test = get_shape_by_axes(bottom_blob, positive_axis, dims - 1);
      if (dim_shape < 0) {
        return -1;
      }
      end = end < dim_shape ? end : dim_shape;
      int cur = start;
      std::vector<int> temp_indice;
      if (step > 0) {
        while (cur < end && cur < dim_shape) {
          temp_indice.push_back(cur);
          cur += step;
        }
      } else if (step < 0) {
        while (cur > end && cur > 0) {
          temp_indice.push_back(cur);
          cur += step;
        }
      } else {
        fprintf(stderr, "step should not be 0!\n");
        return -100;
      }
      active_indice[positive_axis - 1] = temp_indice;
      active_indice[positive_axis - 1].resize(temp_indice.size());
    }
    top_blob.create((int)active_indice[1].size(), (int)active_indice[0].size(), elemsize,
                    opt.blob_allocator);
    for (int i = 0; i < active_indice[0].size(); i++) {
      for (int j = 0; j < active_indice[1].size(); j++) {
        top_blob.row(i)[j] = bottom_blob.row(active_indice[0][i])[active_indice[1][j]];
      }
    }
    return 0;
  }

  if (dims == 3) {
    std::vector<std::vector<int> > active_indice;
    std::vector<int> indices;
    for (int i = 0; i < bottom_blob.c; i++) {
      indices.push_back(i);
    }
    active_indice.push_back(indices);
    indices.clear();
    for (int i = 0; i < bottom_blob.h; i++) {
      indices.push_back(i);
    }
    active_indice.push_back(indices);
    indices.clear();
    for (int i = 0; i < bottom_blob.w; i++) {
      indices.push_back(i);
    }
    active_indice.push_back(indices);
    for (int i = 0; i < axes.w; i++) {
      int positive_axis = axes_ptr[i] < 0 ? dims + axes_ptr[i] : axes_ptr[i];
      int step = step_ptr[i];

      int start = start_ptr[i];
      int end = end_ptr[i];
      int cur = start;
      std::vector<int> temp_indice;
      if (step > 0) {
        while (cur < end && cur < bottom_blob.w) {
          temp_indice.push_back(cur);
          cur += step;
        }
      } else if (step < 0) {
        while (cur > end && cur > 0) {
          temp_indice.push_back(cur);
          cur += step;
        }
      } else {
        fprintf(stderr, "step should not be 0!\n");
        return -100;
      }
      active_indice[positive_axis - 1] = temp_indice;
      active_indice[positive_axis - 1].resize(temp_indice.size());
    }
    top_blob.create((int)active_indice[2].size(), (int)active_indice[1].size(),
                    (int)active_indice[0].size(), elemsize, opt.blob_allocator);
    for (int i = 0; i < active_indice[0].size(); i++) {
      for (int j = 0; j < active_indice[1].size(); j++) {
        for (int k = 0; k < active_indice[2].size(); k++) {
          top_blob.channel(i).row(j)[k] = bottom_blob.channel(active_indice[0][i])
                                              .row(active_indice[1][j])[active_indice[2][k]];
        }
      }
    }
    return 0;
  }

  return 0;
}

}  // namespace mmdeploy
