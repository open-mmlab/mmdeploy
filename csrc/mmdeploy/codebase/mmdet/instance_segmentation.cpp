// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "object_detection.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv_utils.h"

namespace mmdeploy::mmdet
{

    class ResizeInstanceMask : public ResizeBBox
    {
      public:
        explicit ResizeInstanceMask(const Value& cfg)
            : ResizeBBox(cfg)
        {
            if (cfg.contains("params"))
            {
                mask_thr_binary_ = cfg["params"].value("mask_thr_binary", mask_thr_binary_);
                is_rcnn_         = cfg["params"].contains("rcnn");
                is_resize_mask_  = cfg["params"].value("is_resize_mask", is_resize_mask_);
            }
            operation::Context ctx(device_, stream_);
            warp_affine_ = operation::Managed<operation::WarpAffine>::Create("bilinear");
            permute_     = operation::Managed<::mmdeploy::operation::Permute>::Create();
        }
      }

    } else {  // rtmdet-inst
      auto mask_channel = (int)d_mask.shape(0);
      auto mask_height = (int)d_mask.shape(1);
      auto mask_width = (int)d_mask.shape(2);
      // (C, H, W) -> (H, W, C)
      std::vector<int> axes = {1, 2, 0};
      OUTCOME_TRY(permute_.Apply(d_mask, d_mask, axes));
      Device host{"cpu"};
      OUTCOME_TRY(auto cpu_mask, MakeAvailableOnDevice(d_mask, host, stream_));
      OUTCOME_TRY(stream().Wait());
      cv::Mat mask_mat(mask_height, mask_width, CV_32FC(mask_channel), cpu_mask.data());
      int resize_height = int(mask_height / scale_factor_[1] + 0.5);
      int resize_width = int(mask_width / scale_factor_[0] + 0.5);
      // skip resize if scale_factor is 1.0
      if (resize_height != mask_height || resize_width != mask_width) {
        cv::resize(mask_mat, mask_mat, cv::Size(resize_width, resize_height), cv::INTER_LINEAR);
      }
      // crop masks
      mask_mat = mask_mat(cv::Range(0, img_h), cv::Range(0, img_w)).clone();

      for (int i = 0; i < (int)result.size(); i++) {
        cv::Mat mask_;
        cv::extractChannel(mask_mat, mask_, i);
        Tensor mask_t = cpu::CVMat2Tensor(mask_);
        h_warped_masks.emplace_back(mask_t);
      }
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
  ::mmdeploy::operation::Managed<::mmdeploy::operation::Permute> permute_;
  float mask_thr_binary_{.5f};
  bool is_rcnn_{true};
  bool is_resize_mask_{true};
  std::vector<float> scale_factor_{1.0, 1.0, 1.0, 1.0};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMDetection, ResizeInstanceMask);

}  // namespace mmdeploy::mmdet
