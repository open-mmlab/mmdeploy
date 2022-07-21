// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/panet.h"

#include <algorithm>

#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace mmdeploy {

namespace mmocr {

std::vector<std::vector<float>> pixel_group_cpu(const cv::Mat_<float>& score,
                                                const cv::Mat_<uint8_t>& mask,
                                                const cv::Mat_<float>& embedding,
                                                const cv::Mat_<int32_t>& kernel_label,
                                                const cv::Mat_<uint8_t>& kernel_contour,
                                                int kernel_region_num, float dis_threshold);

class PANHead : public MMOCR {
 public:
  explicit PANHead(const Value& config) : MMOCR(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      min_text_avg_confidence_ = params.value("min_text_avg_confidence", min_text_avg_confidence_);
      min_kernel_confidence_ = params.value("min_kernel_confidence", min_kernel_confidence_);
      min_text_avg_confidence_ = params.value("min_text_avg_confidence", min_text_avg_confidence_);
      min_text_area_ = params.value("min_text_area", min_text_area_);
      rescale_ = params.value("rescale", rescale_);
      downsample_ratio_ = params.value("downsample_ratio", downsample_ratio_);
    }
    auto platform = Platform(device_.platform_id()).GetPlatformName();
    auto creator = Registry<PaHeadImpl>::Get().GetCreator(platform);
    if (!creator) {
      MMDEPLOY_ERROR(
          "PANHead: implementation for platform \"{}\" not found. Available platforms: {}",
          platform, Registry<PaHeadImpl>::Get().List());
      throw_exception(eEntryNotFound);
    }
    impl_ = creator->Create(nullptr);
    impl_->Init(stream_);
  }

  Result<Value> operator()(const Value& _data, const Value& _pred) noexcept {
    OUTCOME_TRY(auto pred, MakeAvailableOnDevice(_pred["output"].get<Tensor>(), device_, stream_));
    OUTCOME_TRY(stream_.Wait());

    if (pred.shape().size() != 4 || pred.shape(0) != 1 || pred.data_type() != DataType::kFLOAT) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", pred.shape(),
                     (int)pred.data_type());
      return Status(eNotSupported);
    }

    // drop batch dimension
    pred.Squeeze(0);

    auto text_pred = pred.Slice(0);
    auto kernel_pred = pred.Slice(1);
    auto embed_pred = pred.Slice(2, pred.shape(0));

    cv::Mat_<float> text_score;
    cv::Mat_<uint8_t> text;
    cv::Mat_<uint8_t> kernel;
    cv::Mat_<int> labels;
    cv::Mat_<float> embed;
    int region_num = 0;

    OUTCOME_TRY(impl_->Process(text_pred, kernel_pred, embed_pred, min_text_confidence_,
                               min_kernel_confidence_, text_score, text, kernel, labels, embed,
                               region_num));

    auto text_points = pixel_group_cpu(text_score, text, embed, labels, kernel, region_num,
                                       min_text_avg_confidence_);

    auto scale_w = _data["img_metas"]["scale_factor"][0].get<float>();
    auto scale_h = _data["img_metas"]["scale_factor"][1].get<float>();

    TextDetectorOutput output;
    for (auto& text_point : text_points) {
      auto text_confidence = text_point[0];
      auto area = text_point.size() - 2;
      if (filter_instance(static_cast<float>(area), text_confidence, min_text_area_,
                          min_text_avg_confidence_)) {
        continue;
      }
      cv::Mat_<float> points(text_point.size() / 2 - 1, 2, text_point.data() + 2);
      cv::RotatedRect rect = cv::minAreaRect(points);
      std::vector<cv::Point2f> vertices(4);
      rect.points(vertices.data());
      if (rescale_) {
        for (auto& p : vertices) {
          p.x /= scale_w * downsample_ratio_;
          p.y /= scale_h * downsample_ratio_;
        }
      }
      auto& bbox = output.boxes.emplace_back();
      for (int i = 0; i < 4; ++i) {
        bbox[i * 2] = vertices[i].x;
        bbox[i * 2 + 1] = vertices[i].y;
      }
      output.scores.push_back(text_confidence);
    }
    return to_value(output);
  }

  static bool filter_instance(float area, float confidence, float min_area, float min_confidence) {
    return area < min_area || confidence < min_confidence;
  }

  float min_text_confidence_{.5f};
  float min_kernel_confidence_{.5f};
  float min_text_avg_confidence_{0.85};
  float min_text_area_{16};
  bool rescale_{true};
  float downsample_ratio_{.25f};
  std::unique_ptr<PaHeadImpl> impl_;
};

REGISTER_CODEBASE_COMPONENT(MMOCR, PANHead);

}  // namespace mmocr

MMDEPLOY_DEFINE_REGISTRY(mmocr::PaHeadImpl);

}  // namespace mmdeploy
