// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/psenet.h"

#include <algorithm>
#include <opencv2/opencv.hpp>

#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/utils/device_utils.h"

namespace mmdeploy {
namespace mmocr {

void contour_expand(const cv::Mat_<uint8_t>& kernel_masks, const cv::Mat_<int32_t>& kernel_label,
                    const cv::Mat_<float>& score, int min_kernel_area, int kernel_num,
                    std::vector<int>& text_areas, std::vector<float>& text_scores,
                    std::vector<std::vector<int>>& text_points);

class PSEHead : public MMOCR {
 public:
  explicit PSEHead(const Value& config) : MMOCR(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      min_kernel_confidence_ = params.value("min_kernel_confidence", min_kernel_confidence_);
      min_text_avg_confidence_ = params.value("min_text_avg_confidence", min_text_avg_confidence_);
      min_kernel_area_ = params.value("min_kernel_area", min_kernel_area_);
      min_text_area_ = params.value("min_text_area", min_text_area_);
      rescale_ = params.value("rescale", rescale_);
      downsample_ratio_ = params.value("downsample_ratio", downsample_ratio_);
    }
    auto platform = Platform(device_.platform_id()).GetPlatformName();
    auto creator = Registry<PseHeadImpl>::Get().GetCreator(platform);
    if (!creator) {
      MMDEPLOY_ERROR("PSEHead: implementation for platform \"{}\" not found", platform);
      throw_exception(eEntryNotFound);
    }
    impl_ = creator->Create(nullptr);
    impl_->Init(stream_);
  }

  Result<Value> operator()(const Value& _data, const Value& _pred) noexcept {
    auto _preds = _pred["output"].get<Tensor>();
    if (_preds.shape().size() != 4 || _preds.shape(0) != 1 ||
        _preds.data_type() != DataType::kFLOAT) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", _preds.shape(),
                     (int)_preds.data_type());
      return Status(eNotSupported);
    }

    // drop batch dimension
    _preds.Squeeze(0);

    cv::Mat_<uint8_t> masks;
    cv::Mat_<int> kernel_labels;
    cv::Mat_<float> score;
    int region_num = 0;

    OUTCOME_TRY(
        impl_->Process(_preds, min_kernel_confidence_, score, masks, kernel_labels, region_num));

    std::vector<int> text_areas;
    std::vector<float> text_scores;
    std::vector<std::vector<int>> text_points;
    contour_expand(masks.rowRange(1, masks.rows), kernel_labels, score, min_kernel_area_,
                   region_num, text_areas, text_scores, text_points);

    auto scale_w = _data["img_metas"]["scale_factor"][0].get<float>();
    auto scale_h = _data["img_metas"]["scale_factor"][1].get<float>();

    TextDetectorOutput output;
    for (int text_index = 1; text_index < region_num; ++text_index) {
      auto& text_point = text_points[text_index];
      auto text_confidence = text_scores[text_index];
      auto area = text_areas[text_index];

      if (filter_instance(static_cast<float>(area), text_confidence, min_text_area_,
                          min_text_avg_confidence_)) {
        continue;
      }

      cv::Mat_<int> points(text_point.size() / 2, 2, text_point.data());
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

  float min_kernel_confidence_{.5f};
  float min_text_avg_confidence_{0.85};
  int min_kernel_area_{0};
  float min_text_area_{16};
  bool rescale_{true};
  float downsample_ratio_{.25f};

  std::unique_ptr<PseHeadImpl> impl_;
};

REGISTER_CODEBASE_COMPONENT(MMOCR, PSEHead);

}  // namespace mmocr

MMDEPLOY_DEFINE_REGISTRY(mmocr::PseHeadImpl);

}  // namespace mmdeploy
