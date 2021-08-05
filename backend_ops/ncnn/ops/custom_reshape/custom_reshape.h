#ifndef LAYER_CUSTOMRESHAPE_H
#define LAYER_CUSTOMRESHAPE_H

#include "layer.h"

namespace mmlab {

class CustomReshape : public ncnn::Layer {
 public:
  CustomReshape();

  virtual int load_param(const ncnn::ParamDict& pd);

  virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                      std::vector<ncnn::Mat>& top_blobs,
                      const ncnn::Option& opt) const;

 public:
  // reshape flag
  // 0 = copy from bottom
  // -1 = remaining
  // -233 = drop this dim (default)

  // flag permute chw->hwc or hw->wh before and after reshape
  int permute;
};

}  // namespace mmlab

#endif  // LAYER_CUSTOMRESHAPE_H
