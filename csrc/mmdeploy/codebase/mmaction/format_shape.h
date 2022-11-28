// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMACTION_FORMAT_SHAPE_H_
#define MMDEPLOY_SRC_CODEBASE_MMACTION_FORMAT_SHAPE_H_

#include <array>
#include <vector>

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy {

class FormatShapeImpl : public TransformImpl {
 public:
  explicit FormatShapeImpl(const Value& args);
  ~FormatShapeImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> Format(const std::vector<Tensor>& tensors, int clip_len,
                                int num_clips) = 0;

 protected:
  struct format_shape_arg_t {
    std::string input_format;
  };
  using ArgType = struct format_shape_arg_t;
  ArgType arg_;
};

MMDEPLOY_DECLARE_REGISTRY(FormatShapeImpl, std::unique_ptr<FormatShapeImpl>(const Value& config));

}  // namespace mmdeploy

#endif
