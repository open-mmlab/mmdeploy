// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_IMAGE2TENSOR_H
#define MMDEPLOY_IMAGE2TENSOR_H

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {
/**
 * Convert image to `Tensor` by given keys.
 *
 * The dimension order of input image is (1, H, W, C). The pipeline will convert
 * it to (1, C, H, W).
 *
 */
class MMDEPLOY_API ImageToTensorImpl : public TransformImpl {
 public:
  ImageToTensorImpl(const Value& args);
  ~ImageToTensorImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> HWC2CHW(const Tensor& tensor) = 0;

 protected:
  struct to_img_tensor_arg_t {
    std::vector<std::string> keys;
  };
  using ArgType = struct to_img_tensor_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API ImageToTensor : public Transform {
 public:
  explicit ImageToTensor(const Value& args, int version = 0);
  ~ImageToTensor() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<ImageToTensorImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(ImageToTensorImpl,
                          std::unique_ptr<ImageToTensorImpl>(const Value& config));

}  // namespace mmdeploy

#endif  // MMDEPLOY_IMAGE2TENSOR_H
