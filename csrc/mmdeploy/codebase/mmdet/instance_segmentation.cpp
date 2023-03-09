// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "object_detection.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv_utils.h"

namespace mmdeploy::mmdet {

class ResizeInstanceMask : public ResizeBBox {
 public:
  explicit ResizeInstanceMask(const Value& cfg) : ResizeBBox(cfg) {
    if (cfg.contains("params")) {
      mask_thr_binary_ = cfg["params"].value("mask_thr_binary", mask_thr_binary_);
      is_rcnn_ = cfg["params"].contains("rcnn");
    }
  }

  // TODO: remove duplication
  Result<Value> operator()(const Value& prep_res, const Value& infer_res) {
    MMDEPLOY_DEBUG("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
    try {
      auto dets = infer_res["dets"].get<Tensor>();
      auto labels = infer_res["labels"].get<Tensor>();
      auto masks = infer_res["masks"].get<Tensor>();

      MMDEPLOY_DEBUG("dets.shape: {}", dets.shape());
      MMDEPLOY_DEBUG("labels.shape: {}", labels.shape());
      MMDEPLOY_DEBUG("masks.shape: {}", masks.shape());

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

      if (!(masks.shape().size() == 4 && masks.data_type() == DataType::kFLOAT)) {
        MMDEPLOY_ERROR("unsupported `mask` tensor, shape: {}, dtype: {}", masks.shape(),
                       (int)masks.data_type());
        return Status(eNotSupported);
      }

      OUTCOME_TRY(auto _dets, MakeAvailableOnDevice(dets, kHost, stream()));
      OUTCOME_TRY(auto _labels, MakeAvailableOnDevice(labels, kHost, stream()));
      OUTCOME_TRY(auto _masks, MakeAvailableOnDevice(masks, kHost, stream()));
      OUTCOME_TRY(stream().Wait());

      OUTCOME_TRY(auto result, DispatchGetBBoxes(prep_res["img_metas"], _dets, _labels));

      auto ori_w = prep_res["img_metas"]["ori_shape"][2].get<int>();
      auto ori_h = prep_res["img_metas"]["ori_shape"][1].get<int>();

      ProcessMasks(result, _masks, _dets, ori_w, ori_h);

      return to_value(result);
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("{}", e.what());
      return Status(eFail);
    }
  }

 protected:
  void ProcessMasks(Detections& result, Tensor cpu_masks, Tensor cpu_dets, int img_w,
                    int img_h) const {
    cpu_masks.Squeeze(0);
    cpu_dets.Squeeze(0);
    for (auto& det : result) {
      auto mask = cpu_masks.Slice(det.index);

      cv::Mat mask_mat((int)mask.shape(1), (int)mask.shape(2), CV_32F, mask.data<float>());
      cv::Mat warped_mask;
      auto& bbox = det.bbox;
      // same as mmdet with skip_empty = True
      auto x0 = std::max(std::floor(bbox[0]) - 1, 0.f);
      auto y0 = std::max(std::floor(bbox[1]) - 1, 0.f);
      auto x1 = std::min(std::ceil(bbox[2]) + 1, (float)img_w);
      auto y1 = std::min(std::ceil(bbox[3]) + 1, (float)img_h);
      auto width = static_cast<int>(x1 - x0);
      auto height = static_cast<int>(y1 - y0);
      // params align_corners = False
      float fx;
      float fy;
      float tx;
      float ty;
      if (is_rcnn_) {  // mask r-cnn
        fx = (float)mask_mat.cols / (bbox[2] - bbox[0]);
        fy = (float)mask_mat.rows / (bbox[3] - bbox[1]);
        tx = (x0 + .5f - bbox[0]) * fx - .5f;
        ty = (y0 + .5f - bbox[1]) * fy - .5f;
      } else {  // rtmdet-ins
        auto raw_bbox = cpu_dets.Slice(det.index);
        auto raw_bbox_data = raw_bbox.data<float>();
        fx = (raw_bbox_data[2] - raw_bbox_data[0]) / (bbox[2] - bbox[0]);
        fy = (raw_bbox_data[3] - raw_bbox_data[1]) / (bbox[3] - bbox[1]);
        tx = (x0 + .5f - bbox[0]) * fx - .5f + raw_bbox_data[0];
        ty = (y0 + .5f - bbox[1]) * fy - .5f + raw_bbox_data[1];
      }
      cv::Mat m = (cv::Mat_<float>(2, 3) << fx, 0, tx, 0, fy, ty);
      cv::warpAffine(mask_mat, warped_mask, m, cv::Size{width, height},
                     cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
      warped_mask = warped_mask > mask_thr_binary_;
      det.mask = Mat(height, width, PixelFormat::kGRAYSCALE, DataType::kINT8,
                     std::shared_ptr<void>(warped_mask.data, [mat = warped_mask](void*) {}));
    }
  }

  float mask_thr_binary_{.5f};
  bool is_rcnn_{true};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMDetection, ResizeInstanceMask);

}  // namespace mmdeploy::mmdet
