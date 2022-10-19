// Copyright (c) OpenMMLab. All rights reserved.
#ifndef MMDEPLOY_CODEBASE_MMDET_SSD_HEAD_H_
#define MMDEPLOY_CODEBASE_MMDET_SSD_HEAD_H_

#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/core/tensor.h"

#define NUM_RESULTS         1917
#define NUM_CLASS 91
#define Y_SCALE  10.0f
#define X_SCALE  10.0f
#define H_SCALE  5.0f
#define W_SCALE  5.0f

namespace mmdeploy::mmdet {

class SSDHead : public MMDetection {
 public:
  explicit SSDHead(const Value& cfg);

  Result<Value> operator()(const Value& prep_res, const Value& infer_res);
  std::vector<Tensor> GetDetsLabels(const Value& prep_res, const Value& infer_res);

 private:
  static constexpr int NUM_SIZE = 4;
  std::vector<std::vector<float>> priors_;
};
} // namespace mmdeploy::mmdet

#endif // MMDEPLOY_CODEBASE_MMDET_SSD_HEAD_H_
