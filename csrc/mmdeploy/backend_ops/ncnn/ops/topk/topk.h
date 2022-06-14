// Copyright (c) OpenMMLab. All rights reserved.
#ifndef LAYER_TOPK_H
#define LAYER_TOPK_H

#include "layer.h"

namespace mmdeploy {

class TopK : public ncnn::Layer {
 public:
  TopK();
  virtual int load_param(const ncnn::ParamDict& pd);
  virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs,
                      const ncnn::Option& opt) const;

 public:
  int axis;
  int largest;
  int sorted;
  int keep_dims;
};

}  // namespace mmdeploy

#endif  // LAYER_TOPK_H
