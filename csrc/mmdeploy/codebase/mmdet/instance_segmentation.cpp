// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
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
    operation::Context ctx(device_, stream_);
    warp_affine_ = operation::Managed<operation::WarpAffine>::Create("bilinear");
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
      // Note: `masks` are kept on device to avoid data copy overhead from device to host.
      // refer to https://github.com/open-mmlab/mmdeploy/issues/1849
      // OUTCOME_TRY(auto _masks, MakeAvailableOnDevice(masks, kHost, stream()));
      // OUTCOME_TRY(stream().Wait());

      OUTCOME_TRY(auto result, DispatchGetBBoxes(prep_res["img_metas"], _dets, _labels));

      auto ori_w = prep_res["img_metas"]["ori_shape"][2].get<int>();
      auto ori_h = prep_res["img_metas"]["ori_shape"][1].get<int>();

      ProcessMasks(result, masks, _dets, ori_w, ori_h);

      return to_value(result);
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("{}", e.what());
      return Status(eFail);
    }
  }

 protected:
  Result<void> ProcessMasks(Detections& result, Tensor d_mask, Tensor cpu_dets, int img_w,
                            int img_h) {
    d_mask.Squeeze(0);
    cpu_dets.Squeeze(0);

    ::mmdeploy::operation::Context ctx(device_, stream_);

    std::vector<Tensor> warped_masks;
    warped_masks.reserve(result.size());

    std::vector<Tensor> h_warped_masks;
    h_warped_masks.reserve(result.size());

    for (auto& det : result) {
      auto mask = d_mask.Slice(det.index);
      auto mask_height = (int)mask.shape(1);
      auto mask_width = (int)mask.shape(2);
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
        fx = (float)mask_width / (bbox[2] - bbox[0]);
        fy = (float)mask_height / (bbox[3] - bbox[1]);
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

      float affine_matrix[] = {fx, 0, tx, 0, fy, ty};

      cv::Mat_<float> m(2, 3, affine_matrix);
      cv::invertAffineTransform(m, m);

      mask.Reshape({1, mask_height, mask_width, 1});

      Tensor& warped_mask = warped_masks.emplace_back();
      OUTCOME_TRY(warp_affine_.Apply(mask, warped_mask, affine_matrix, height, width));

      OUTCOME_TRY(CopyToHost(warped_mask, h_warped_masks.emplace_back()));
    }

    OUTCOME_TRY(stream_.Wait());

    for (size_t i = 0; i < h_warped_masks.size(); ++i) {
      result[i].mask = ThresholdMask(h_warped_masks[i]);
    }

    return success();
  }

  Result<void> CopyToHost(const Tensor& src, Tensor& dst) {
    if (src.device() == kHost) {
      dst = src;
      return success();
    }
    dst = TensorDesc{kHost, src.data_type(), src.shape()};
    OUTCOME_TRY(stream_.Copy(src.buffer(), dst.buffer(), dst.byte_size()));
    return success();
  }

  Mat ThresholdMask(const Tensor& h_mask) const {
    cv::Mat warped_mat = cpu::Tensor2CVMat(h_mask);
    warped_mat = warped_mat > mask_thr_binary_;
    return {warped_mat.rows, warped_mat.cols, PixelFormat::kGRAYSCALE, DataType::kINT8,
            std::shared_ptr<void>(warped_mat.data, [mat = warped_mat](void*) {})};
  }

 private:
  operation::Managed<operation::WarpAffine> warp_affine_;
  float mask_thr_binary_{.5f};
  bool is_rcnn_{true};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMDetection, ResizeInstanceMask);

}  // namespace mmdeploy::mmdet
