// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CROP_H
#define MMDEPLOY_CROP_H

#include "core/tensor.h"
#include "transform.h"

namespace mmdeploy {

class CenterCropImpl : public TransformImpl {
 public:
  explicit CenterCropImpl(const Value& args);
  ~CenterCropImpl() = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                                   int right) = 0;

 protected:
  struct center_crop_arg_t {
    std::array<int, 2> crop_size;
  };
  using ArgType = struct center_crop_arg_t;

 protected:
  ArgType arg_;
};

class CenterCrop : public Transform {
 public:
  explicit CenterCrop(const Value& args, int version = 0);
  ~CenterCrop() = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 protected:
  std::unique_ptr<CenterCropImpl> impl_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CROP_H
