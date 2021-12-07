// Copyright (c) OpenMMLab. All rights reserved.
#ifndef LAYER_GATHER_H
#define LAYER_GATHER_H

#include "layer.h"

namespace mmdeploy {

class Gather : public ncnn::Layer {
 public:
  Gather();

  virtual int load_param(const ncnn::ParamDict& pd);

  virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs,
                      const ncnn::Option& opt) const;

 public:
  int axis;
};

}  // namespace mmdeploy

#endif  // LAYER_GATHER_H
