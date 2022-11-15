// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_RESIZE_H
#define MMDEPLOY_RESIZE_H

#include <array>

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {
class MMDEPLOY_API ResizeImpl : public TransformImpl {
 public:
  explicit ResizeImpl(const Value& args);
  ~ResizeImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> ResizeImage(const Tensor& src_img, int dst_h, int dst_w) = 0;

 protected:
  struct resize_arg_t {
    std::array<int, 2> img_scale;
    std::string interpolation{"bilinear"};
    bool keep_ratio{true};
  };
  using ArgType = resize_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API Resize : public Transform {
 public:
  explicit Resize(const Value& args, int version = 0);
  ~Resize() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<ResizeImpl> impl_;
  static const std::string name_;
};

MMDEPLOY_DECLARE_REGISTRY(ResizeImpl);

}  // namespace mmdeploy
#endif  // MMDEPLOY_RESIZE_H
