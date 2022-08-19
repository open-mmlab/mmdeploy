// Copyright (c) OpenMMLab. All rights reserved.
#include "shape.h"

#include "../ncnn_ops_definer.h"

namespace mmdeploy {
using namespace ncnn;
DEFINE_LAYER_CREATOR(Shape)
DEFINE_NCNN_OPS(Shape, Shape)
Shape::Shape() {
  one_blob_only = true;
  support_inplace = false;
}

int Shape::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const {
  int dims = bottom_blob.dims;
  int w = bottom_blob.w;
  size_t elemsize = sizeof(float);
  top_blob.create(dims + 1, elemsize, opt.blob_allocator);
  if (top_blob.empty()) {
    return -100;
  }
  float *outptr = top_blob;

  if (dims == 1) {
    outptr[0] = 1.0f;
    outptr[1] = w;
  } else if (dims == 2) {
    int h = bottom_blob.h;
    outptr[0] = 1.0f;
    outptr[1] = h;
    outptr[2] = w;
  } else if (dims == 3) {
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    outptr[0] = 1.0f;
    outptr[1] = channels;
    outptr[2] = h;
    outptr[3] = w;
  } else {
    fprintf(stdout, "Unsupported dims=%d\n", dims);
  }

  return 0;
}

}  // namespace mmdeploy
