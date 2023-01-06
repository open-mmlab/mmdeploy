// Copyright (c) OpenMMLab. All rights reserved.
#include "rtmdet_head.h"

#include <math.h>

#include <algorithm>
#include <numeric>

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "utils.h"

namespace mmdeploy::mmdet {

RTMDetSepBNHead::RTMDetSepBNHead(const Value& cfg) : MMDetection(cfg) {
  auto init = [&]() -> Result<void> {
    auto model = cfg["context"]["model"].get<Model>();
    if (cfg.contains("params")) {
      nms_pre_ = cfg["params"].value("nms_pre", -1);
      score_thr_ = cfg["params"].value("score_thr", 0.02f);
      min_bbox_size_ = cfg["params"].value("min_bbox_size", 0);
      max_per_img_ = cfg["params"].value("max_per_img", 100);
      iou_threshold_ = cfg["params"].contains("nms")
                           ? cfg["params"]["nms"].value("iou_threshold", 0.45f)
                           : 0.45f;
      if (cfg["params"].contains("anchor_generator")) {
        offset_ = cfg["params"]["anchor_generator"].value("offset", 0);
        from_value(cfg["params"]["anchor_generator"]["strides"], strides_);
      }
    }
    return success();
  };
  init().value();
}

Result<Value> RTMDetSepBNHead::operator()(const Value& prep_res, const Value& infer_res) {
  MMDEPLOY_DEBUG("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
  try {
    std::vector<Tensor> cls_scores;
    std::vector<Tensor> bbox_preds;
    const Device kHost{0, 0};
    int i = 0;
    int divisor = infer_res.size() / 2;
    for (auto iter = infer_res.begin(); iter != infer_res.end(); iter++) {
      auto pred_map = iter->get<Tensor>();
      OUTCOME_TRY(auto _pred_map, MakeAvailableOnDevice(pred_map, kHost, stream()));
      if (i < divisor)
        cls_scores.push_back(_pred_map);
      else
        bbox_preds.push_back(_pred_map);
      i++;
    }
    OUTCOME_TRY(stream().Wait());
    OUTCOME_TRY(auto result, GetBBoxes(prep_res["img_metas"], bbox_preds, cls_scores));
    return to_value(result);
  } catch (...) {
    return Status(eFail);
  }
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

Result<Detections> RTMDetSepBNHead::GetBBoxes(const Value& prep_res,
                                              const std::vector<Tensor>& bbox_preds,
                                              const std::vector<Tensor>& cls_scores) const {
  MMDEPLOY_DEBUG("bbox_pred: {}, {}", bbox_preds[0].shape(), dets[0].data_type());
  MMDEPLOY_DEBUG("cls_score: {}, {}", scores[0].shape(), scores[0].data_type());

  std::vector<float> filter_boxes;
  std::vector<float> obj_probs;
  std::vector<int> class_ids;

  for (int i = 0; i < bbox_preds.size(); i++) {
    RTMDetFeatDeocde(bbox_preds[i], cls_scores[i], strides_[i], offset_, filter_boxes, obj_probs,
                     class_ids);
  }

  std::vector<int> indexArray;
  for (int i = 0; i < obj_probs.size(); ++i) {
    indexArray.push_back(i);
  }
  Sort(obj_probs, class_ids, indexArray);

  Tensor dets(TensorDesc{Device{0, 0}, DataType::kFLOAT,
                         TensorShape{int(filter_boxes.size() / 4), 4}, "dets"});
  std::copy(filter_boxes.begin(), filter_boxes.end(), dets.data<float>());
  NMS(dets, iou_threshold_, indexArray);

  Detections objs;
  std::vector<float> scale_factor;
  if (prep_res.contains("scale_factor")) {
    from_value(prep_res["scale_factor"], scale_factor);
  } else {
    scale_factor = {1.f, 1.f, 1.f, 1.f};
  }
  int ori_width = prep_res["ori_shape"][2].get<int>();
  int ori_height = prep_res["ori_shape"][1].get<int>();
  auto det_ptr = dets.data<float>();
  for (int i = 0; i < indexArray.size(); ++i) {
    if (indexArray[i] == -1) {
      continue;
    }
    int j = indexArray[i];
    auto x1 = det_ptr[j * 4 + 0];
    auto y1 = det_ptr[j * 4 + 1];
    auto x2 = det_ptr[j * 4 + 2];
    auto y2 = det_ptr[j * 4 + 3];
    int label_id = class_ids[i];
    float score = obj_probs[i];

    MMDEPLOY_DEBUG("{}-th box: ({}, {}, {}, {}), {}, {}", i, x1, y1, x2, y2, label_id, score);

    auto rect =
        MapToOriginImage(x1, y1, x2, y2, scale_factor.data(), 0, 0, ori_width, ori_height, 0, 0);
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

int RTMDetSepBNHead::RTMDetFeatDeocde(const Tensor& bbox_pred, const Tensor& cls_score,
                                      const float stride, const float offset,
                                      std::vector<float>& filter_boxes,
                                      std::vector<float>& obj_probs,
                                      std::vector<int>& class_ids) const {
  int cls_param_num = cls_score.shape(1);
  int feat_h = bbox_pred.shape(2);
  int feat_w = bbox_pred.shape(3);
  int feat_size = feat_h * feat_w;
  auto bbox_ptr = bbox_pred.data<float>();
  auto score_ptr = cls_score.data<float>();  // (b, c, h, w)
  int valid_count = 0;
  for (int i = 0; i < feat_h; i++) {
    for (int j = 0; j < feat_w; j++) {
      float max_score = score_ptr[i * feat_w + j];
      int class_id = 0;
      for (int k = 0; k < cls_param_num; k++) {
        float score = score_ptr[k * feat_size + i * feat_w + j];
        if (score > max_score) {
          max_score = score;
          class_id = k;
        }
      }
      max_score = sigmoid(max_score);
      if (max_score < score_thr_) continue;

      obj_probs.push_back(max_score);
      class_ids.push_back(class_id);

      float tl_x = bbox_ptr[0 * feat_size + i * feat_w + j];
      float tl_y = bbox_ptr[1 * feat_size + i * feat_w + j];
      float br_x = bbox_ptr[2 * feat_size + i * feat_w + j];
      float br_y = bbox_ptr[3 * feat_size + i * feat_w + j];

      auto box = RTMDetdecode(tl_x, tl_y, br_x, br_y, stride, offset, j, i);

      tl_x = box[0];
      tl_y = box[1];
      br_x = box[2];
      br_y = box[3];

      filter_boxes.push_back(tl_x);
      filter_boxes.push_back(tl_y);
      filter_boxes.push_back(br_x);
      filter_boxes.push_back(br_y);
      valid_count++;
    }
  }
  return valid_count;
}

std::array<float, 4> RTMDetSepBNHead::RTMDetdecode(float tl_x, float tl_y, float br_x, float br_y,
                                                   float stride, float offset, int j, int i) const {
  tl_x = (offset + j) * stride - tl_x;
  tl_y = (offset + i) * stride - tl_y;
  br_x = (offset + j) * stride + br_x;
  br_y = (offset + i) * stride + br_y;
  return std::array<float, 4>{tl_x, tl_y, br_x, br_y};
}

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMDetection, RTMDetSepBNHead);

}  // namespace mmdeploy::mmdet
