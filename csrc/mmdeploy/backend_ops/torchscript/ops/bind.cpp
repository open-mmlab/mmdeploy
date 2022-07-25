// Copyright (c) OpenMMLab. All rights reserved.
#include "torch/script.h"

TORCH_LIBRARY(mmdeploy, m) {
  m.def(
       "modulated_deform_conv(Tensor input, Tensor weight, Tensor bias, Tensor offset, Tensor "
       "mask, "
       "int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int "
       "dilation_h,int dilation_w, int groups, int deform_groups, bool with_bias) -> Tensor")
      .def(
          "coreml_nms(Tensor boxes, Tensor scores, float iou_threshold, "
          "float score_threshold, int max_boxes) -> Tensor[]");
}
