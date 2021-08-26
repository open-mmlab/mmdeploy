#ifndef LAYER_SHAPE_H
#define LAYER_SHAPE_H

#include "layer.h"

namespace mmlab {

class Shape : public ncnn::Layer {
 public:
  Shape();

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                      const ncnn::Option& opt) const;
};

}  // namespace mmlab

#endif  // LAYER_SHAPE_H
