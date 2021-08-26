#ifndef LAYER_TOPK_H
#define LAYER_TOPK_H

#include "layer.h"

namespace mmlab {

class TopK : public ncnn::Layer {
 public:
  TopK();
  virtual int load_param(const ncnn::ParamDict& pd);
  virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                      std::vector<ncnn::Mat>& top_blobs,
                      const ncnn::Option& opt) const;

 public:
  int axis;
  int largest;
  int sorted;
};

}  // namespace mmlab

#endif  // LAYER_TOPK_H
