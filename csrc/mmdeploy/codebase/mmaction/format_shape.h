// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMACTION_FORMAT_SHAPE_H_
#define MMDEPLOY_CODEBASE_MMACTION_FORMAT_SHAPE_H_

#include <array>
#include <string>
#include <vector>

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::mmaction {

class FormatShapeImpl : public operation::Operation {
 public:
  explicit FormatShapeImpl(const std::string_view& input_format);

  Result<void> apply(const std::vector<Tensor>& inputs, Tensor& output, int clip_len,
                     int num_clips);

  Result<Tensor> FormatNCHW(Tensor& src, int clip_len, int num_clips);

  Result<Tensor> FormatNCTHW(Tensor& src, int clip_len, int num_clips);

  Result<void> MergeInputs(const std::vector<Tensor>& images, Tensor& inputs);

 protected:
  std::string input_format_;
  operation::Managed<operation::Permute> permute_;
};

class FormatShape : public Transform {
 public:
  explicit FormatShape(const Value& args);

  Result<void> Apply(Value& data) override;

 private:
  operation::Managed<FormatShapeImpl> format_;
};

MMDEPLOY_DECLARE_REGISTRY(FormatShapeImpl,
                          std::unique_ptr<FormatShapeImpl>(std::string input_format));

}  // namespace mmdeploy::mmaction

#endif
