// Copyright (c) OpenMMLab. All rights reserved.
#ifndef LAYER_CONSTANTOFSHAPE_H
#define LAYER_CONSTANTOFSHAPE_H

#include "layer.h"

namespace mmdeploy {

class ConstantOfShape : public ncnn::Layer {
 public:
  ConstantOfShape();

  virtual int load_param(const ncnn::ParamDict& pd);

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                      const ncnn::Option& opt) const;

 public:
  float val;
};

}  // namespace mmdeploy

#endif  // LAYER_CONSTANTOFSHAPE_H
