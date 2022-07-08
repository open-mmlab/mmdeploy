// Copyright (c) OpenMMLab. All rights reserved.
#include "constantofshape.h"

#include "../ncnn_ops_definer.h"

namespace mmdeploy {
using namespace ncnn;
DEFINE_LAYER_CREATOR(ConstantOfShape)
DEFINE_NCNN_OPS(ConstantOfShape, ConstantOfShape)
ConstantOfShape::ConstantOfShape() {
  one_blob_only = true;
  support_inplace = false;
}

int ConstantOfShape::load_param(const ParamDict& pd) {
  val = pd.get(0, 0.f);
  return 0;
}

int ConstantOfShape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
  int dims = bottom_blob.w - 1;
  const float* bottom_ptr = bottom_blob;
  const float* shape_ptr = bottom_ptr + 1;

  if (dims == 1) {
    int w = (int)(shape_ptr[0] + 0.5);
    size_t elemsize = sizeof(val);
    top_blob.create(w, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;
    top_blob.fill(val);
    return 0;
  } else if (dims == 2) {
    int h = (int)(shape_ptr[0] + 0.5);
    int w = (int)(shape_ptr[1] + 0.5);
    size_t elemsize = sizeof(val);
    top_blob.create(w, h, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;
    top_blob.fill(val);
    return 0;
  } else if (dims == 3) {
    int channels = (int)(shape_ptr[0] + 0.5);
    int h = (int)(shape_ptr[1] + 0.5);
    int w = (int)(shape_ptr[2] + 0.5);
    size_t elemsize = sizeof(val);
    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;
    top_blob.fill(val);
    return 0;
  }
  return -1;
}

}  // namespace mmdeploy
