// Copyright (c) OpenMMLab. All rights reserved.
#include "utils.h"

#include <algorithm>
#include <numeric>

using mmdeploy::framework::Tensor;

namespace mmdeploy::mmdet {

std::array<float, 4> MapToOriginImage(float left, float top, float right, float bottom,
                                      const float* scale_factor, float x_offset, float y_offset,
                                      int ori_width, int ori_height) {
  left = std::max(left / scale_factor[0] + x_offset, 0.f);
  top = std::max(top / scale_factor[1] + y_offset, 0.f);
  right = std::min(right / scale_factor[2] + x_offset, (float)ori_width - 1.f);
  bottom = std::min(bottom / scale_factor[3] + y_offset, (float)ori_height - 1.f);
  return {left, top, right, bottom};
}

void FilterScoresAndTopk(const Tensor& scores, float score_thr, int topk, std::vector<float>& probs,
                         std::vector<int>& label_ids, std::vector<int>& anchor_idxs) {
  auto kDets = scores.shape(1);
  auto kClasses = scores.shape(2);
  auto score_ptr = scores.data<float>();

  for (auto i = 0; i < kDets; ++i, score_ptr += kClasses) {
    auto iter = std::max_element(score_ptr, score_ptr + kClasses);
    auto max_score = *iter;
    if (*iter < score_thr) {
      continue;
    }
    probs.push_back(*iter);
    label_ids.push_back(iter - score_ptr);
    anchor_idxs.push_back(i);
  }
}

float IOU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
          float ymax1) {
  auto w = std::max(0.f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1));
  auto h = std::max(0.f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1));
  auto area = w * h;
  auto sum = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1);
  auto iou = area / (sum - area);
  return iou <= 0.f ? 0.f : iou;
}

void NMS(const Tensor& dets, float iou_threshold, std::vector<int>& keep_idxs) {
  auto det_ptr = dets.data<float>();
  for (auto i = 0; i < keep_idxs.size(); ++i) {
    auto n = keep_idxs[i];
    for (auto j = i + 1; j < keep_idxs.size(); ++j) {
      auto m = keep_idxs[j];

      // `delta_xywh_bbox_coder` decode return tl_x, tl_y, br_x, br_y
      float xmin0 = det_ptr[n * 4 + 0];
      float ymin0 = det_ptr[n * 4 + 1];
      float xmax0 = det_ptr[n * 4 + 2];
      float ymax0 = det_ptr[n * 4 + 3];

      float xmin1 = det_ptr[m * 4 + 0];
      float ymin1 = det_ptr[m * 4 + 1];
      float xmax1 = det_ptr[m * 4 + 2];
      float ymax1 = det_ptr[m * 4 + 3];

      float iou = IOU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > iou_threshold) {
        keep_idxs[j] = -1;
      }
    }
  }
}

void Sort(std::vector<float>& probs, std::vector<int>& label_ids, std::vector<int>& anchor_idxs) {
  std::vector<int> prob_idxs(probs.size());
  std::iota(prob_idxs.begin(), prob_idxs.end(), 0);
  std::sort(prob_idxs.begin(), prob_idxs.end(), [&](int i, int j) { return probs[i] > probs[j]; });
  std::vector<float> _probs;
  std::vector<int> _label_ids;
  std::vector<int> _keep_idxs;
  for (auto idx : prob_idxs) {
    _probs.push_back(probs[idx]);
    _label_ids.push_back(label_ids[idx]);
    _keep_idxs.push_back(anchor_idxs[idx]);
  }
  probs = std::move(_probs);
  label_ids = std::move(_label_ids);
  anchor_idxs = std::move(_keep_idxs);
}

}  // namespace mmdeploy::mmdet
