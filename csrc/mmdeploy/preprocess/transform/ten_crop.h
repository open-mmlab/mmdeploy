// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TEN_CROP_H
#define MMDEPLOY_TEN_CROP_H

#include <array>

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {

class MMDEPLOY_API TenCropImpl : public TransformImpl {
 public:
  explicit TenCropImpl(const Value& args);
  ~TenCropImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                                   int right) = 0;
  virtual Result<Tensor> HorizontalFlip(const Tensor& tensor) = 0;

 protected:
  struct ten_crop_arg_t {
    std::array<int, 2> crop_size;
  };
  using ArgType = struct ten_crop_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API TenCrop : public Transform {
 public:
  explicit TenCrop(const Value& args, int version = 0);
  ~TenCrop() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 protected:
  std::unique_ptr<TenCropImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(TenCropImpl, std::unique_ptr<TenCropImpl>(const Value& config));

}  // namespace mmdeploy

#endif
