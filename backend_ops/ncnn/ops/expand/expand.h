#ifndef LAYER_EXPAND_H
#define LAYER_EXPAND_H

#include "layer.h"

namespace mmlab {

class Expand : public ncnn::Layer {
 public:
  Expand();

  virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                      std::vector<ncnn::Mat>& top_blobs,
                      const ncnn::Option& opt) const;
};

}  // namespace mmlab

#endif  // LAYER_EXPAND_H
