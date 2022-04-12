// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_

#include "pipeline2.h"

namespace mmdeploy::async {

class InferenceParser {
 public:
  static Result<unique_ptr<Pipeline>> Parse(const Value& config);
};

}  // namespace mmdeploy::async

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_
