// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_THREE_CROP_H
#define MMDEPLOY_THREE_CROP_H

#include <array>

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {

class MMDEPLOY_API ThreeCropImpl : public TransformImpl {
 public:
  explicit ThreeCropImpl(const Value& args);
  ~ThreeCropImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                                   int right) = 0;

 protected:
  struct three_crop_arg_t {
    std::array<int, 2> crop_size;
  };
  using ArgType = struct three_crop_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API ThreeCrop : public Transform {
 public:
  explicit ThreeCrop(const Value& args, int version = 0);
  ~ThreeCrop() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 protected:
  std::unique_ptr<ThreeCropImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(ThreeCropImpl, std::unique_ptr<ThreeCropImpl>(const Value& config));

}  // namespace mmdeploy

#endif
