// Copyright (c) OpenMMLab. All rights reserved.

#include <opencv2/imgproc.hpp>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmrotate.h"
#include "opencv_utils.h"

namespace mmdeploy::mmrotate {

using std::vector;

class ResizeRBBox : public MMRotate {
 public:
  explicit ResizeRBBox(const Value& cfg) : MMRotate(cfg) {
    if (cfg.contains("params")) {
      score_thr_ = cfg["params"].value("score_thr", 0.05f);
    }
  }

  Result<Value> operator()(const Value& prep_res, const Value& infer_res) {
    MMDEPLOY_DEBUG("prep_res: {}", prep_res);
    MMDEPLOY_DEBUG("infer_res: {}", infer_res);

    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto dets,
                MakeAvailableOnDevice(infer_res["dets"].get<Tensor>(), cpu_device, stream_));
    OUTCOME_TRY(auto labels,
                MakeAvailableOnDevice(infer_res["labels"].get<Tensor>(), cpu_device, stream_));
    OUTCOME_TRY(stream_.Wait());

    if (!(dets.shape().size() == 3 && dets.shape(2) == 6 && dets.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `dets` tensor, shape: {}, dtype: {}", dets.shape(),
                     (int)dets.data_type());
      return Status(eNotSupported);
    }

    if (labels.shape().size() != 2) {
      MMDEPLOY_ERROR("unsupported `labels`, tensor, shape: {}, dtype: {}", labels.shape(),
                     (int)labels.data_type());
      return Status(eNotSupported);
    }

    OUTCOME_TRY(auto result, DispatchGetBBoxes(prep_res["img_metas"], dets, labels));
    return to_value(result);
  }

  Result<RotatedDetectorOutput> DispatchGetBBoxes(const Value& prep_res, const Tensor& dets,
                                                  const Tensor& labels) {
    auto data_type = labels.data_type();
    switch (data_type) {
      case DataType::kFLOAT:
        return GetRBBoxes<float>(prep_res, dets, labels);
      case DataType::kINT32:
        return GetRBBoxes<int32_t>(prep_res, dets, labels);
      case DataType::kINT64:
        return GetRBBoxes<int64_t>(prep_res, dets, labels);
      default:
        return Status(eNotSupported);
    }
  }

  template <typename T>
  Result<RotatedDetectorOutput> GetRBBoxes(const Value& prep_res, const Tensor& dets,
                                           const Tensor& labels) {
    RotatedDetectorOutput objs;
    auto* dets_ptr = dets.data<float>();
    auto* labels_ptr = labels.data<T>();
    vector<float> scale_factor;
    if (prep_res.contains("scale_factor")) {
      from_value(prep_res["scale_factor"], scale_factor);
    } else {
      scale_factor = {1.f, 1.f, 1.f, 1.f};
    }

    int ori_width = prep_res["ori_shape"][2].get<int>();
    int ori_height = prep_res["ori_shape"][1].get<int>();

    auto bboxes_number = dets.shape()[1];
    auto channels = dets.shape()[2];
    for (auto i = 0; i < bboxes_number; ++i, dets_ptr += channels, ++labels_ptr) {
      float score = dets_ptr[channels - 1];
      if (score <= score_thr_) {
        continue;
      }
      auto cx = dets_ptr[0] / scale_factor[0];
      auto cy = dets_ptr[1] / scale_factor[1];
      auto width = dets_ptr[2] / scale_factor[0];
      auto height = dets_ptr[3] / scale_factor[1];
      auto angle = dets_ptr[4];
      RotatedDetectorOutput::Detection det{};
      det.label_id = static_cast<int>(*labels_ptr);
      det.score = score;
      det.rbbox = {cx, cy, width, height, angle};
      objs.detections.push_back(std::move(det));
    }

    return objs;
  }

 private:
  float score_thr_;
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMRotate, ResizeRBBox);

}  // namespace mmdeploy::mmrotate
