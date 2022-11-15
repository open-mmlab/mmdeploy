// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_LOAD_H
#define MMDEPLOY_LOAD_H

#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {
class MMDEPLOY_API PrepareImageImpl : public TransformImpl {
 public:
  explicit PrepareImageImpl(const Value& args);
  ~PrepareImageImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> ConvertToBGR(const Mat& img) = 0;
  virtual Result<Tensor> ConvertToGray(const Mat& img) = 0;

 protected:
  struct prepare_image_arg_t {
    bool to_float32{false};
    std::string color_type{"color"};
  };
  using ArgType = struct prepare_image_arg_t;

  ArgType arg_;
};

class MMDEPLOY_API PrepareImage : public Transform {
 public:
  explicit PrepareImage(const Value& args, int version = 0);
  ~PrepareImage() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<PrepareImageImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(PrepareImageImpl);

}  // namespace mmdeploy

#endif  // MMDEPLOY_LOAD_H
