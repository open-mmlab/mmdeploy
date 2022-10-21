// Copyright (c) OpenMMLab. All rights reserved.
#ifndef MMDEPLOY_CODEBASE_MMDET_SSD_HEAD_H_
#define MMDEPLOY_CODEBASE_MMDET_SSD_HEAD_H_

#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/core/tensor.h"

namespace mmdeploy::mmdet {

class SSDHead : public MMDetection {
 public:
  explicit SSDHead(const Value& cfg);

  Result<Value> operator()(const Value& prep_res, const Value& infer_res);
  Result<Detections> GetBBoxes(const Value& prep_res, const Value& infer_res);

 private:
  // @brief Filter results using score threshold and topk candidates.
  // scores (Tensor): The scores, shape (num_bboxes, K).
  // probs: The scores after being filtered
  // label_ids: The class labels
  // anchor_idxs: The anchor indexes
  static void FilterScoresAndTopk(Tensor& scores, float score_thr, int topk,
                                  std::vector<float>& probs, std::vector<int>& label_ids,
                                  std::vector<int>& anchor_idxs);
  static float IOU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1,
                   float xmax1, float ymax1);

  static void Sort(std::vector<float>& probs, std::vector<int>& label_ids,
                   std::vector<int>& anchor_idxs);

  static void NMS(Tensor& dets, float iou_threshold, std::vector<int>& keep_idxs);

 private:
  float score_thr_{0.4f};
  int nms_pre_{1000};
  float iou_threshold_{0.45f};
  int min_bbox_size_ {0};
};
}  // namespace mmdeploy::mmdet

#endif  // MMDEPLOY_CODEBASE_MMDET_SSD_HEAD_H_
