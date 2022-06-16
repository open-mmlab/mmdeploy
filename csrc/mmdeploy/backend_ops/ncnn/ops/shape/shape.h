// Copyright (c) OpenMMLab. All rights reserved.
#ifndef LAYER_SHAPE_H
#define LAYER_SHAPE_H

#include "layer.h"

namespace mmdeploy {

class Shape : public ncnn::Layer {
 public:
  Shape();

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                      const ncnn::Option& opt) const;
};

}  // namespace mmdeploy

#endif  // LAYER_SHAPE_H
