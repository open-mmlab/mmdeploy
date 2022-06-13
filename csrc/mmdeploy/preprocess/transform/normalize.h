// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_NORMALIZE_H
#define MMDEPLOY_NORMALIZE_H

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {

class MMDEPLOY_API NormalizeImpl : public TransformImpl {
 public:
  explicit NormalizeImpl(const Value& args);
  ~NormalizeImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> NormalizeImage(const Tensor& img) = 0;

 protected:
  struct normalize_arg_t {
    std::vector<float> mean;
    std::vector<float> std;
    bool to_rgb;
  };
  using ArgType = struct normalize_arg_t;
  ArgType arg_;
};

class MMDEPLOY_API Normalize : public Transform {
 public:
  explicit Normalize(const Value& args, int version = 0);
  ~Normalize() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<NormalizeImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(NormalizeImpl);

}  // namespace mmdeploy
#endif  // MMDEPLOY_NORMALIZE_H
