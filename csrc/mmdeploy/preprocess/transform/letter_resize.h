// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_RESIZE_H
#define MMDEPLOY_RESIZE_H

#include <array>

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {
class MMDEPLOY_API LetterResizeImpl : public TransformImpl {
 public:
  explicit LetterResizeImpl(const Value& args);
  ~LetterResizeImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> ResizeImage(const Tensor& src_img, int dst_h, int dst_w) = 0;
  virtual Result<Tensor> PadImage(const Tensor& img, const int& top, const int& left,
                                  const int& bottom, const int& right, const float& pad_val) = 0;

 protected:
  struct resize_arg_t {
    std::array<int, 2> img_scale;
    std::string interpolation{"bilinear"};
    float pad_val{0};
    bool keep_ratio{true};
    bool use_mini_pad{false};
    bool stretch_only{false};
    bool allow_scale_up{true};
  };
  using ArgType = resize_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API LetterResize : public Transform {
 public:
  explicit LetterResize(const Value& args, int version = 0);
  ~LetterResize() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<LetterResizeImpl> impl_;
  static const std::string name_;
};

MMDEPLOY_DECLARE_REGISTRY(LetterResizeImpl);

}  // namespace mmdeploy
#endif  // MMDEPLOY_RESIZE_H
