// Copyright (c) OpenMMLab. All rights reserved.
#include "yolo_head.h"

#include <math.h>

#include <algorithm>
#include <numeric>

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "utils.h"

namespace mmdeploy::mmdet {

YOLOHead::YOLOHead(const Value& cfg) : MMDetection(cfg) {
  auto init = [&]() -> Result<void> {
    auto model = cfg["context"]["model"].get<Model>();
    if (cfg.contains("params")) {
      nms_pre_ = cfg["params"].value("nms_pre", -1);
      score_thr_ = cfg["params"].value("score_thr", 0.02f);
      min_bbox_size_ = cfg["params"].value("min_bbox_size", 0);
      iou_threshold_ = cfg["params"].contains("nms")
                           ? cfg["params"]["nms"].value("iou_threshold", 0.45f)
                           : 0.45f;
      if (cfg["params"].contains("anchor_generator")) {
        from_value(cfg["params"]["anchor_generator"]["base_sizes"], anchors_);
        from_value(cfg["params"]["anchor_generator"]["strides"], strides_);
      }
    }
    return success();
  };
  init().value();
}

Result<Value> YOLOHead::operator()(const Value& prep_res, const Value& infer_res) {
  MMDEPLOY_DEBUG("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
  try {
    const Device kHost{0, 0};
    std::vector<Tensor> pred_maps;
    for (auto iter = infer_res.begin(); iter != infer_res.end(); iter++) {
      auto pred_map = iter->get<Tensor>();
      OUTCOME_TRY(auto _pred_map, MakeAvailableOnDevice(pred_map, kHost, stream()));
      pred_maps.push_back(_pred_map);
    }
    OUTCOME_TRY(stream().Wait());
    // reorder pred_maps according to strides and anchors, mainly for rknpu yolov3
    if ((pred_maps.size() > 1) &&
        !((strides_[0] < strides_[1]) ^ (pred_maps[0].shape(3) < pred_maps[1].shape(3)))) {
      std::reverse(pred_maps.begin(), pred_maps.end());
    }
    OUTCOME_TRY(auto result, GetBBoxes(prep_res["img_metas"], pred_maps));
    return to_value(result);
  } catch (...) {
    return Status(eFail);
  }
}

inline static int clamp(float val, int min, int max) {
  return val > min ? (val < max ? val : max) : min;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

int YOLOHead::YOLOFeatDecode(const Tensor& feat_map, const std::vector<std::vector<float>>& anchor,
                             int grid_h, int grid_w, int height, int width, int stride,
                             std::vector<float>& boxes, std::vector<float>& obj_probs,
                             std::vector<int>& class_id, float threshold) const {
  auto input = const_cast<float*>(feat_map.data<float>());
  auto prop_box_size = feat_map.shape(1) / anchor.size();
  const int kClasses = prop_box_size - 5;
  int valid_count = 0;
  int grid_len = grid_h * grid_w;
  float thres = unsigmoid(threshold);
  for (int a = 0; a < anchor.size(); a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        float box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres) {
          int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
          float* in_ptr = input + offset;

          float box_x = sigmoid(*in_ptr);
          float box_y = sigmoid(in_ptr[grid_len]);
          float box_w = in_ptr[2 * grid_len];
          float box_h = in_ptr[3 * grid_len];
          auto box = yolo_decode(box_x, box_y, box_w, box_h, stride, anchor, j, i, a);

          box_x = box[0];
          box_y = box[1];
          box_w = box[2];
          box_h = box[3];

          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);
          boxes.push_back(box_x);
          boxes.push_back(box_y);
          boxes.push_back(box_x + box_w);
          boxes.push_back(box_y + box_h);

          float max_class_probs = in_ptr[5 * grid_len];
          int max_class_id = 0;
          for (int k = 1; k < kClasses; ++k) {
            float prob = in_ptr[(5 + k) * grid_len];
            if (prob > max_class_probs) {
              max_class_id = k;
              max_class_probs = prob;
            }
          }
          obj_probs.push_back(sigmoid(max_class_probs) * sigmoid(box_confidence));
          class_id.push_back(max_class_id);
          valid_count++;
        }
      }
    }
  }
  return valid_count;
}

Result<Detections> YOLOHead::GetBBoxes(const Value& prep_res,
                                       const std::vector<Tensor>& pred_maps) const {
  std::vector<float> filter_boxes;
  std::vector<float> obj_probs;
  std::vector<int> class_id;

  int model_in_h = prep_res["img_shape"][1].get<int>();
  int model_in_w = prep_res["img_shape"][2].get<int>();

  for (int i = 0; i < pred_maps.size(); i++) {
    int stride = strides_[i];
    int grid_h = model_in_h / stride;
    int grid_w = model_in_w / stride;
    YOLOFeatDecode(pred_maps[i], anchors_[i], grid_h, grid_w, model_in_h, model_in_w, stride,
                   filter_boxes, obj_probs, class_id, score_thr_);
  }

  std::vector<int> indexArray;
  for (int i = 0; i < obj_probs.size(); ++i) {
    indexArray.push_back(i);
  }
  Sort(obj_probs, class_id, indexArray);

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
    auto x1 = clamp(det_ptr[j * 4 + 0], 0, model_in_w);
    auto y1 = clamp(det_ptr[j * 4 + 1], 0, model_in_h);
    auto x2 = clamp(det_ptr[j * 4 + 2], 0, model_in_w);
    auto y2 = clamp(det_ptr[j * 4 + 3], 0, model_in_h);
    int label_id = class_id[i];
    float score = obj_probs[i];

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

Result<Value> YOLOV3Head::operator()(const Value& prep_res, const Value& infer_res) {
  return YOLOHead::operator()(prep_res, infer_res);
}

std::array<float, 4> YOLOV3Head::yolo_decode(float box_x, float box_y, float box_w, float box_h,
                                             float stride,
                                             const std::vector<std::vector<float>>& anchor, int j,
                                             int i, int a) const {
  box_x = (box_x + j) * stride;
  box_y = (box_y + i) * stride;
  box_w = expf(box_w) * anchor[a][0];
  box_h = expf(box_h) * anchor[a][1];
  return std::array<float, 4>{box_x, box_y, box_w, box_h};
}

Result<Value> YOLOV5Head::operator()(const Value& prep_res, const Value& infer_res) {
  return YOLOHead::operator()(prep_res, infer_res);
}

std::array<float, 4> YOLOV5Head::yolo_decode(float box_x, float box_y, float box_w, float box_h,
                                             float stride,
                                             const std::vector<std::vector<float>>& anchor, int j,
                                             int i, int a) const {
  box_x = box_x * 2 - 0.5;
  box_y = box_y * 2 - 0.5;
  box_w = box_w * 2 - 0.5;
  box_h = box_h * 2 - 0.5;
  box_x = (box_x + j) * stride;
  box_y = (box_y + i) * stride;
  box_w = box_w * box_w * anchor[a][0];
  box_h = box_h * box_h * anchor[a][1];
  return std::array<float, 4>{box_x, box_y, box_w, box_h};
}

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMDetection, YOLOV3Head);
MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMDetection, YOLOV5Head);

}  // namespace mmdeploy::mmdet
