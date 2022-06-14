// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_DEFAULT_FORMAT_BUNDLE_H
#define MMDEPLOY_DEFAULT_FORMAT_BUNDLE_H

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {
/**
 * It simplifies the pipeline of formatting common fields
 */
class MMDEPLOY_API DefaultFormatBundleImpl : public TransformImpl {
 public:
  DefaultFormatBundleImpl(const Value& args);
  ~DefaultFormatBundleImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> ToFloat32(const Tensor& tensor, const bool& img_to_float) = 0;
  virtual Result<Tensor> HWC2CHW(const Tensor& tensor) = 0;

 protected:
  struct default_format_bundle_arg_t {
    bool img_to_float = true;
  };
  using ArgType = struct default_format_bundle_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API DefaultFormatBundle : public Transform {
 public:
  explicit DefaultFormatBundle(const Value& args, int version = 0);
  ~DefaultFormatBundle() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<DefaultFormatBundleImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(DefaultFormatBundleImpl);

}  // namespace mmdeploy

#endif  // MMDEPLOY_DEFAULT_FORMAT_BUNDLE_H
