// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMDET_UTILS_H_
#define MMDEPLOY_CODEBASE_MMDET_UTILS_H_

#include <array>
#include <vector>

#include "mmdeploy/core/tensor.h"

namespace mmdeploy::mmdet {
std::array<float, 4> MapToOriginImage(float left, float top, float right, float bottom,
                                      const float* scale_factor, float x_offset, float y_offset,
                                      int ori_width, int ori_height, int top_padding,
                                      int left_padding);
// @brief Filter results using score threshold and topk candidates.
// scores (Tensor): The scores, shape (num_bboxes, K).
// probs: The scores after being filtered
// label_ids: The class labels
// anchor_idxs: The anchor indexes
void FilterScoresAndTopk(const mmdeploy::framework::Tensor& scores, float score_thr, int topk,
                         std::vector<float>& probs, std::vector<int>& label_ids,
                         std::vector<int>& anchor_idxs);
float IOU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
          float ymax1);

void Sort(std::vector<float>& probs, std::vector<int>& label_ids, std::vector<int>& anchor_idxs);

void NMS(const mmdeploy::framework::Tensor& dets, float iou_threshold, std::vector<int>& keep_idxs);

}  // namespace mmdeploy::mmdet

#endif  // MMDEPLOY_CODEBASE_MMDET_UTILS_H_
