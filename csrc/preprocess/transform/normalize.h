// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_NORMALIZE_H
#define MMDEPLOY_NORMALIZE_H

#include "core/tensor.h"
#include "transform.h"

namespace mmdeploy {

class NormalizeImpl : public TransformImpl {
 public:
  explicit NormalizeImpl(const Value& args);
  ~NormalizeImpl() = default;

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

class Normalize : public Transform {
 public:
  explicit Normalize(const Value& args, int version = 0);
  ~Normalize() = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<NormalizeImpl> impl_;
};

}  // namespace mmdeploy
#endif  // MMDEPLOY_NORMALIZE_H
