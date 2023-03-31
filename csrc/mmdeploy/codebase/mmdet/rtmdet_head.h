// Copyright (c) OpenMMLab. All rights reserved.
#ifndef MMDEPLOY_CODEBASE_MMDET_RTMDET_HEAD_H_
#define MMDEPLOY_CODEBASE_MMDET_RTMDET_HEAD_H_

#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/core/tensor.h"

namespace mmdeploy::mmdet {

class RTMDetSepBNHead : public MMDetection {
 public:
  explicit RTMDetSepBNHead(const Value& cfg);
  Result<Value> operator()(const Value& prep_res, const Value& infer_res);
  Result<Detections> GetBBoxes(const Value& prep_res, const std::vector<Tensor>& bbox_preds,
                               const std::vector<Tensor>& cls_scores) const;
  int RTMDetFeatDeocde(const Tensor& bbox_pred, const Tensor& cls_score, const float stride,
                       const float offset, std::vector<float>& filter_boxes,
                       std::vector<float>& obj_probs, std::vector<int>& class_ids) const;
  std::array<float, 4> RTMDetdecode(float tl_x, float tl_y, float br_x, float br_y, float stride,
                                    float offset, int j, int i) const;

 private:
  float score_thr_{0.4f};
  int nms_pre_{1000};
  float iou_threshold_{0.45f};
  int min_bbox_size_{0};
  int max_per_img_{100};
  float offset_{0.0f};
  std::vector<float> strides_;
};

}  // namespace mmdeploy::mmdet

#endif  // MMDEPLOY_CODEBASE_MMDET_RTMDET_HEAD_H_
