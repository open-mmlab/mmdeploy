// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_PAD_H
#define MMDEPLOY_PAD_H

#include <array>

#include "mmdeploy/core/tensor.h"
#include "transform.h"

namespace mmdeploy {

class MMDEPLOY_API PadImpl : public TransformImpl {
 public:
  explicit PadImpl(const Value& args);
  ~PadImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  // pad on all sides with specified padding mode
  // @param padding: Padding on each border in clock-wise (left, top, right,
  // bottom)
  virtual Result<Tensor> PadImage(const Tensor& img, const std::array<int, 4>& padding) = 0;

 protected:
  struct pad_arg_t {
    std::array<int, 2> size;
    int size_divisor;
    float pad_val;
    bool pad_to_square;
    std::string padding_mode;
  };
  using ArgType = struct pad_arg_t;
  ArgType arg_;
};

class MMDEPLOY_API Pad : public Transform {
 public:
  explicit Pad(const Value& args, int version = 0);
  ~Pad() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 protected:
  std::unique_ptr<PadImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(PadImpl);

}  // namespace mmdeploy

#endif  // MMDEPLOY_PAD_H
