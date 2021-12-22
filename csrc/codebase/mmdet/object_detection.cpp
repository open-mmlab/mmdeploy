// Copyright (c) OpenMMLab. All rights reserved.

#include "object_detection.h"

#include "core/registry.h"
#include "core/utils/device_utils.h"
#include "experimental/module_adapter.h"

using namespace std;

namespace mmdeploy::mmdet {

ResizeBBox::ResizeBBox(const Value& cfg) : MMDetection(cfg) {
  if (cfg.contains("params")) {
    score_thr_ = cfg["params"].value("score_thr", 0.f);
    min_bbox_size_ = cfg["params"].value("min_bbox_size", 0.f);
  }
}
Result<Value> ResizeBBox::operator()(const Value& prep_res, const Value& infer_res) {
  DEBUG("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
  try {
    auto dets = infer_res["dets"].get<Tensor>();
    auto labels = infer_res["labels"].get<Tensor>();

    DEBUG("dets.shape: {}", dets.shape());
    DEBUG("labels.shape: {}", labels.shape());

    // `dets` is supposed to have 3 dims. They are 'batch', 'bboxes_number'
    // and 'channels' respectively
    if (!(dets.shape().size() == 3 && dets.data_type() == DataType::kFLOAT)) {
      ERROR("unsupported `dets` tensor, shape: {}, dtype: {}", dets.shape(), (int)dets.data_type());
      return Status(eNotSupported);
    }

    // `labels` is supposed to have 2 dims, which are 'batch' and
    // 'bboxes_number'
    if (labels.shape().size() != 2) {
      ERROR("unsupported `labels`, tensor, shape: {}, dtype: {}", labels.shape(),
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
Result<DetectorOutput> ResizeBBox::DispatchGetBBoxes(const Value& prep_res, const Tensor& dets,
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
Result<DetectorOutput> ResizeBBox::GetBBoxes(const Value& prep_res, const Tensor& dets,
                                             const Tensor& labels) {
  DetectorOutput objs;
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

    DEBUG("ori left {}, top {}, right {}, bottom {}, label {}", left, top, right, bottom,
          *labels_ptr);
    auto rect = MapToOriginImage(left, top, right, bottom, scale_factor.data(), w_offset, h_offset,
                                 ori_width, ori_height);
    if (rect[2] - rect[0] < min_bbox_size_ || rect[3] - rect[1] < min_bbox_size_) {
      DEBUG("ignore small bbox with width '{}' and height '{}", rect[2] - rect[0],
            rect[3] - rect[1]);
      continue;
    }
    DEBUG("remap left {}, top {}, right {}, bottom {}", rect.left, rect.top, rect.right,
          rect.bottom);
    DetectorOutput::Detection det{};
    det.index = i;
    det.label_id = static_cast<int>(*labels_ptr);
    det.score = score;
    det.bbox = rect;
    objs.detections.push_back(std::move(det));
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

REGISTER_CODEBASE_COMPONENT(MMDetection, ResizeBBox);

}  // namespace mmdeploy::mmdet
