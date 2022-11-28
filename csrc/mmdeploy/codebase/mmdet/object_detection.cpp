// Copyright (c) OpenMMLab. All rights reserved.

#include "object_detection.h"

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"

using namespace std;

namespace mmdeploy::mmdet {

ResizeBBox::ResizeBBox(const Value& cfg) : MMDetection(cfg) {
  if (cfg.contains("params")) {
    if (cfg["params"].contains("conf_thr")) {
      // for mobilev2yolov3
      score_thr_ = cfg["params"].value("conf_thr", 0.f);
    } else {
      score_thr_ = cfg["params"].value("score_thr", 0.f);
    }
    min_bbox_size_ = cfg["params"].value("min_bbox_size", 0.f);
  }
}
std::vector<Tensor> ResizeBBox::GetDetsLabels(const Value& prep_res, const Value& infer_res) {
  std::vector<Tensor> results;
  if (infer_res.contains("dets") && infer_res.contains("labels")) {
    results.push_back(infer_res["dets"].get<Tensor>());
    results.push_back(infer_res["labels"].get<Tensor>());
    return results;
  } else if (infer_res.contains("detection_output") && (!infer_res.contains("dets")) &&
             (!infer_res.contains("labels"))) {
    int img_width = prep_res["img_metas"]["img_shape"][2].get<int>();
    int img_height = prep_res["img_metas"]["img_shape"][1].get<int>();
    auto detection_output = infer_res["detection_output"].get<Tensor>();
    auto* detection_output_ptr = detection_output.data<float>();
    // detection_output: (1, num_det, 6)
    TensorDesc labeldesc = detection_output.desc();
    int batch_size = detection_output.shape()[0];
    int num_det = detection_output.shape()[1];
    labeldesc.shape = {batch_size, num_det};
    Tensor labels(labeldesc);
    TensorDesc detdesc = detection_output.desc();
    detdesc.shape = {batch_size, num_det, 5};
    Tensor dets(detdesc);
    auto* dets_ptr = dets.data<float>();
    auto* labels_ptr = labels.data<float>();

    for (int i = 0; i < batch_size * num_det; ++i) {
      *labels_ptr++ = detection_output_ptr[0] - 1;
      dets_ptr[4] = detection_output_ptr[1];
      dets_ptr[0] = detection_output_ptr[2] * img_width;
      dets_ptr[1] = detection_output_ptr[3] * img_height;
      dets_ptr[2] = detection_output_ptr[4] * img_width;
      dets_ptr[3] = detection_output_ptr[5] * img_height;
      dets_ptr += 5;
      detection_output_ptr += 6;
    }
    results.push_back(dets);
    results.push_back(labels);
    return results;
  } else {
    MMDEPLOY_ERROR("No support for another key of detection results!");
    return results;
  }
}
Result<Value> ResizeBBox::operator()(const Value& prep_res, const Value& infer_res) {
  MMDEPLOY_DEBUG("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
  try {
    Tensor dets, labels;
    vector<Tensor> outputs = GetDetsLabels(prep_res, infer_res);
    dets = outputs[0];
    labels = outputs[1];
    MMDEPLOY_DEBUG("dets.shape: {}", dets.shape());
    MMDEPLOY_DEBUG("labels.shape: {}", labels.shape());
    // `dets` is supposed to have 3 dims. They are 'batch', 'bboxes_number'
    // and 'channels' respectively
    if (!(dets.shape().size() == 3 && dets.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `dets` tensor, shape: {}, dtype: {}", dets.shape(),
                     (int)dets.data_type());
      return Status(eNotSupported);
    }

    // `labels` is supposed to have 2 dims, which are 'batch' and
    // 'bboxes_number'
    if (labels.shape().size() != 2) {
      MMDEPLOY_ERROR("unsupported `labels`, tensor, shape: {}, dtype: {}", labels.shape(),
                     (int)labels.data_type());
      return Status(eNotSupported);
    }

    OUTCOME_TRY(auto _dets, MakeAvailableOnDevice(dets, kHost, stream()));
    OUTCOME_TRY(auto _labels, MakeAvailableOnDevice(labels, kHost, stream()));
    OUTCOME_TRY(stream().Wait());

    OUTCOME_TRY(auto result, DispatchGetBBoxes(prep_res["img_metas"], _dets, _labels));
    return to_value(result);

  } catch (...) {
    return Status(eFail);
  }
}
Result<Detections> ResizeBBox::DispatchGetBBoxes(const Value& prep_res, const Tensor& dets,
                                                 const Tensor& labels) {
  auto data_type = labels.data_type();
  switch (data_type) {
    case DataType::kFLOAT:
      return GetBBoxes<float>(prep_res, dets, labels);
    case DataType::kINT32:
      return GetBBoxes<int32_t>(prep_res, dets, labels);
    case DataType::kINT64:
      return GetBBoxes<int64_t>(prep_res, dets, labels);
    default:
      return Status(eNotSupported);
  }
}
template <typename T>
Result<Detections> ResizeBBox::GetBBoxes(const Value& prep_res, const Tensor& dets,
                                         const Tensor& labels) {
  Detections objs;
  auto* dets_ptr = dets.data<float>();
  auto* labels_ptr = labels.data<T>();
  vector<float> scale_factor;
  if (prep_res.contains("scale_factor")) {
    from_value(prep_res["scale_factor"], scale_factor);
  } else {
    scale_factor = {1.f, 1.f, 1.f, 1.f};
  }

  float w_offset = 0.f;
  float h_offset = 0.f;
  int ori_width = prep_res["ori_shape"][2].get<int>();
  int ori_height = prep_res["ori_shape"][1].get<int>();

  // `dets` has shape(1, n, 4) or shape(1, n, 5). The latter one has `score`
  auto bboxes_number = dets.shape()[1];
  auto channels = dets.shape()[2];
  for (auto i = 0; i < bboxes_number; ++i, dets_ptr += channels, ++labels_ptr) {
    float score = 0.f;
    if (channels > 4 && dets_ptr[4] <= score_thr_) {
      continue;
    }
    score = channels > 4 ? dets_ptr[4] : score;
    auto left = dets_ptr[0];
    auto top = dets_ptr[1];
    auto right = dets_ptr[2];
    auto bottom = dets_ptr[3];

    MMDEPLOY_DEBUG("ori left {}, top {}, right {}, bottom {}, label {}", left, top, right, bottom,
                   *labels_ptr);
    auto rect = MapToOriginImage(left, top, right, bottom, scale_factor.data(), w_offset, h_offset,
                                 ori_width, ori_height);
    if (rect[2] - rect[0] < min_bbox_size_ || rect[3] - rect[1] < min_bbox_size_) {
      MMDEPLOY_DEBUG("ignore small bbox with width '{}' and height '{}", rect[2] - rect[0],
                     rect[3] - rect[1]);
      continue;
    }
    MMDEPLOY_DEBUG("remap left {}, top {}, right {}, bottom {}", rect[0], rect[1], rect[2],
                   rect[3]);
    Detection det{};
    det.index = i;
    det.label_id = static_cast<int>(*labels_ptr);
    det.score = score;
    det.bbox = rect;
    objs.push_back(std::move(det));
  }
  return objs;
}
std::array<float, 4> ResizeBBox::MapToOriginImage(float left, float top, float right, float bottom,
                                                  const float* scale_factor, float x_offset,
                                                  float y_offset, int ori_width, int ori_height) {
  left = std::max(left / scale_factor[0] + x_offset, 0.f);
  top = std::max(top / scale_factor[1] + y_offset, 0.f);
  right = std::min(right / scale_factor[2] + x_offset, (float)ori_width - 1.f);
  bottom = std::min(bottom / scale_factor[3] + y_offset, (float)ori_height - 1.f);
  return {left, top, right, bottom};
}

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMDetection, ResizeBBox);

}  // namespace mmdeploy::mmdet
