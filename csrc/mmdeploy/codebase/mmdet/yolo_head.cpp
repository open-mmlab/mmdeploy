// Copyright (c) OpenMMLab. All rights reserved.
#include "yolo_head.h"

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
      if(cfg["params"].contains("anchor_generator")){
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
    for(auto iter = infer_res.begin(); iter != infer_res.end(); iter++)
    {
      auto pred_map = iter->get<Tensor>();
      OUTCOME_TRY(auto _pred_map, MakeAvailableOnDevice(pred_map, kHost, stream()));
      pred_maps.push_back(_pred_map);
    }
    OUTCOME_TRY(stream().Wait());
    OUTCOME_TRY(auto result, GetBBoxes(prep_res["img_metas"], pred_maps));
    return to_value(result);
  } catch (...) {
    return Status(eFail);
  }
}

inline static int clamp(float val, int min, int max) {
  return val > min ? (val < max ? val : max) : min;
}

Result<Detections> YOLOHead::GetBBoxes(const Value& prep_res, const std::vector<Tensor>& pred_maps) const {
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  int model_in_h = prep_res["img_shape"][1].get<int>();
  int model_in_w =  prep_res["img_shape"][2].get<int>();

  for(int i=0; i< pred_maps.size(); i++){
    int stride = strides_[i];
    int grid_h = model_in_h / stride;
    int grid_w = model_in_w / stride;
    YOLOV3FeatDecode(pred_maps[i], anchors_[i], grid_h, grid_w,model_in_h, model_in_w, stride, filterBoxes, objProbs, classId, score_thr_);
  }

  std::vector<int> indexArray;
  for (int i = 0; i < objProbs.size(); ++i) {
    indexArray.push_back(i);
  }
  Sort(objProbs, classId, indexArray);

  Tensor dets(TensorDesc{Device{0, 0}, DataType::kFLOAT, TensorShape{int(filterBoxes.size()/4), 4}, "dets"});
  memcpy(dets.data<float>(), (float*)filterBoxes.data(), filterBoxes.size() * sizeof(float));
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
    int label_id = classId[i];
    float score = objProbs[i];

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

REGISTER_CODEBASE_COMPONENT(MMDetection, YOLOHead);

}  // namespace mmdeploy::mmdet
