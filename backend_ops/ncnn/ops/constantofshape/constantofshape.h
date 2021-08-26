#ifndef LAYER_CONSTANTOFSHAPE_H
#define LAYER_CONSTANTOFSHAPE_H

#include "layer.h"

namespace mmlab {

class ConstantOfShape : public ncnn::Layer {
 public:
  ConstantOfShape();

  virtual int load_param(const ncnn::ParamDict& pd);

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                      const ncnn::Option& opt) const;

 public:
  float val;
};

}  // namespace mmlab

#endif  // LAYER_CONSTANTOFSHAPE_H
