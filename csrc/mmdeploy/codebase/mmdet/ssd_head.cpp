#include "ssd_head.h"

#include <numeric>

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "utils.h"

namespace mmdeploy::mmdet {

void SSDHead::FilterScoresAndTopk(Tensor& scores, float score_thr, int topk,
                                  std::vector<float>& probs, std::vector<int>& label_ids,
                                  std::vector<int>& anchor_idxs) {
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

float SSDHead::IOU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1,
                   float xmax1, float ymax1) {
  auto w = std::max(0.f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1));
  auto h = std::max(0.f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1));
  auto area = w * h;
  auto sum = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1);
  auto iou = area / (sum - area);
  return iou <= 0.f ? 0.f : iou;
}

void SSDHead::NMS(Tensor& dets, float iou_threshold, std::vector<int>& keep_idxs) {
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

void SSDHead::Sort(std::vector<float>& probs, std::vector<int>& label_ids,
                   std::vector<int>& anchor_idxs) {
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

SSDHead::SSDHead(const Value& cfg) : MMDetection(cfg) {
  auto init = [&]() -> Result<void> {
    auto model = cfg["context"]["model"].get<Model>();
    //    OUTCOME_TRY(auto str_priors, model.ReadFile("box_priors.txt"));
    //    std::istringstream ss(str_priors);

    //    priors_.reserve(NUM_SIZE);
    //    for (int i = 0; i < NUM_SIZE; ++i) {
    //      std::vector<float> prior(NUM_RESULTS);
    //      for (int j = 0; j < NUM_RESULTS; ++j) {
    //        ss >> prior[j];
    //      }
    //      priors_.push_back(prior);
    //    }
    return success();
  };
  init().value();
}

Result<Value> SSDHead::operator()(const Value& prep_res, const Value& infer_res) {
  MMDEPLOY_INFO("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
  try {
    OUTCOME_TRY(auto result, GetBBoxes(prep_res, infer_res));
    return to_value(result);
  } catch (...) {
    return Status(eFail);
  }
}

Result<Detections> SSDHead::GetBBoxes(const Value& prep_res, const Value& infer_res) {
  auto dets = infer_res["dets"].get<Tensor>();
  auto scores = infer_res["labels"].get<Tensor>();

  MMDEPLOY_INFO("dets: {}, {}", dets.shape(), dets.data_type());
  MMDEPLOY_INFO("scores: {}, {}", scores.shape(), scores.data_type());

  std::vector<float> probs;
  std::vector<int> label_ids;
  std::vector<int> anchor_idxs;

  FilterScoresAndTopk(scores, score_thr_, nms_pre_, probs, label_ids, anchor_idxs);

  Sort(probs, label_ids, anchor_idxs);

  NMS(dets, iou_threshold_, anchor_idxs);

  Detections objs;
  std::vector<float> scale_factor;
  if (prep_res.contains("scale_factor")) {
    from_value(prep_res["scale_factor"], scale_factor);
  } else {
    scale_factor = {1.f, 1.f, 1.f, 1.f};
  }
  int ori_width = prep_res["img_metas"]["ori_shape"][2].get<int>();
  int ori_height = prep_res["img_metas"]["ori_shape"][1].get<int>();
  auto det_ptr = dets.data<float>();
  for (int i = 0; i < anchor_idxs.size(); ++i) {
    if (anchor_idxs[i] == -1) {
      continue;
    }
    int j = anchor_idxs[i];
    int x1 = (int)(det_ptr[j * 4 + 0]);
    int y1 = (int)(det_ptr[j * 4 + 1]);
    int x2 = (int)(det_ptr[j * 4 + 2]);
    int y2 = (int)(det_ptr[j * 4 + 3]);
    int label_id = label_ids[i];
    float score = probs[i];

    MMDEPLOY_INFO("{}-th box: ({}, {}, {}, {}), {}, {}", i, x1, y1, x2, y2, label_id, score);

    auto rect = MapToOriginImage(x1, y1, x2, y2, scale_factor.data(), 0, 0, ori_width, ori_height);
    if (rect[2] - rect[0] < min_bbox_size_ || rect[3] - rect[1] < min_bbox_size_) {
      MMDEPLOY_DEBUG("ignore small bbox with width '{}' and height '{}", rect[2] - rect[0],
                     rect[3] - rect[1]);
      continue;
    }
    Detection det{};
    det.index = i;
    det.label_id = label_id;
    det.score = score;
    det.bbox = rect;
    objs.push_back(std::move(det));
  }

  return objs;
}

REGISTER_CODEBASE_COMPONENT(MMDetection, SSDHead);

}  // namespace mmdeploy::mmdet
