// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_MMDET_OBJECT_DETECTION_H_
#define MMDEPLOY_SRC_CODEBASE_MMDET_OBJECT_DETECTION_H_

#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy::mmdet {

class ResizeBBox : public MMDetection {
 public:
  explicit ResizeBBox(const Value& cfg);

  Result<Value> operator()(const Value& prep_res, const Value& infer_res);

 protected:
  Result<DetectorOutput> DispatchGetBBoxes(const Value& prep_res, const Tensor& dets,
                                           const Tensor& labels);

  template <typename T>
  Result<DetectorOutput> GetBBoxes(const Value& prep_res, const Tensor& dets, const Tensor& labels);

  std::array<float, 4> MapToOriginImage(float left, float top, float right, float bottom,
                                        const float* scale_factor, float x_offset, float y_offset,
                                        int ori_width, int ori_height);

  std::vector<Tensor> GetDetsLabels(const Value& prep_res, const Value& infer_res);

 protected:
  constexpr static Device kHost{0, 0};
  float score_thr_{0.f};
  float min_bbox_size_{0.f};
};

}  // namespace mmdeploy::mmdet

#endif  // MMDEPLOY_SRC_CODEBASE_MMDET_OBJECT_DETECTION_H_
