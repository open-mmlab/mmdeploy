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

class FormatShape : public Transform {
 public:
  explicit FormatShape(const Value& args);

  Result<void> Apply(Value& data) override;

  Result<void> Format(const std::vector<Tensor>& images, Tensor& output, int clip_len,
                      int num_clips);

  Result<void> FormatNCHW(Tensor& src, int clip_len, int num_clips, Tensor& dst);

  Result<void> FormatNCTHW(Tensor& src, int clip_len, int num_clips, Tensor& dst);

  Result<void> MergeInputs(const std::vector<Tensor>& images, Tensor& inputs);

 private:
  std::string input_format_;
  operation::Managed<operation::Permute> permute_;
};

}  // namespace mmdeploy::mmaction

#endif
