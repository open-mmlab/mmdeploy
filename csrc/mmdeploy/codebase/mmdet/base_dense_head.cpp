#include "base_dense_head.h"

#include <numeric>

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "utils.h"

namespace mmdeploy::mmdet {

BaseDenseHead::BaseDenseHead(const Value& cfg) : MMDetection(cfg) {
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

Result<Value> BaseDenseHead::operator()(const Value& prep_res, const Value& infer_res) {
  MMDEPLOY_DEBUG("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
  try {
    OUTCOME_TRY(stream_.Wait());
    OUTCOME_TRY(auto result, GetBBoxes(prep_res, infer_res));
    return to_value(result);
  } catch (...) {
    return Status(eFail);
  }
}

Result<Detections> BaseDenseHead::GetBBoxes(const Value& prep_res, const Value& infer_res) const {
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
  if (prep_res["img_metas"].contains("scale_factor")) {
    from_value(prep_res["img_metas"]["scale_factor"], scale_factor);
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
    auto x1 = det_ptr[j * 4 + 0];
    auto y1 = det_ptr[j * 4 + 1];
    auto x2 = det_ptr[j * 4 + 2];
    auto y2 = det_ptr[j * 4 + 3];
    int label_id = label_ids[i];
    float score = probs[i];

    MMDEPLOY_DEBUG("{}-th box: ({}, {}, {}, {}), {}, {}", i, x1, y1, x2, y2, label_id, score);

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

REGISTER_CODEBASE_COMPONENT(MMDetection, BaseDenseHead);

}  // namespace mmdeploy::mmdet
