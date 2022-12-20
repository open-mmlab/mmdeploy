// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMACTION_FORMAT_SHAPE_H_
#define MMDEPLOY_SRC_CODEBASE_MMACTION_FORMAT_SHAPE_H_

#include <array>
#include <vector>

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::mmaction {

class FormatShapeOp : public operation::Operation {
 public:
  explicit FormatShapeOp(std::string input_format) : input_format_(std::move(input_format)){};

  Result<void> apply(const std::vector<Tensor>& inputs, Tensor& output, int clip_len,
                     int num_clips);

  virtual const Device& GetDevice() = 0;

  virtual Result<Tensor> Transpose(Tensor& src, const TensorShape& src_dims,
                                   const std::vector<int>& permutation) = 0;

  Result<Tensor> FormatNCHW(Tensor& src, int clip_len, int num_clips);

  Result<Tensor> FormatNCTHW(Tensor& src, int clip_len, int num_clips);

  Result<void> MergeInputs(const std::vector<Tensor>& images, Tensor& inputs);

 protected:
  std::string input_format_;
};

class FormatShape : public Transform {
 public:
  explicit FormatShape(const Value& args);

  Result<void> Apply(Value& data) override;

 private:
  operation::Managed<FormatShapeOp> format_;
};

MMDEPLOY_DECLARE_REGISTRY(FormatShapeOp, std::unique_ptr<FormatShapeOp>(std::string input_format));

}  // namespace mmdeploy::mmaction

#endif
