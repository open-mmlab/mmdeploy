// Copyright (c) OpenMMLab. All rights reserved.
#ifndef MMDEPLOY_CODEBASE_MMDET_BASE_DENSE_HEAD_H_
#define MMDEPLOY_CODEBASE_MMDET_BASE_DENSE_HEAD_H_

#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/core/tensor.h"

namespace mmdeploy::mmdet {

class BaseDenseHead : public MMDetection {
 public:
  explicit BaseDenseHead(const Value& cfg);

  Result<Value> operator()(const Value& prep_res, const Value& infer_res);
  Result<Detections> GetBBoxes(const Value& prep_res, const Tensor& dets,
                               const Tensor& scores) const;

 private:
  float score_thr_{0.4f};
  int nms_pre_{1000};
  float iou_threshold_{0.45f};
  int min_bbox_size_{0};
};
}  // namespace mmdeploy::mmdet

#endif  // MMDEPLOY_CODEBASE_MMDET_BASE_DENSE_HEAD_H_
