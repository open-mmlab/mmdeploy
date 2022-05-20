// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <opencv2/opencv.hpp>

#include "codebase/mmocr/mmocr.h"
#include "core/device.h"
#include "core/registry.h"
#include "core/serialization.h"
#include "core/utils/device_utils.h"

namespace mmdeploy::mmocr {

void contour_expand(const cv::Mat_<uint8_t>& kernel_masks, const cv::Mat_<int32_t>& kernel_label,
                    const cv::Mat_<float>& score, int min_kernel_area, int kernel_num,
                    std::vector<int>& text_areas, std::vector<float>& text_scores,
                    std::vector<std::vector<int>>& text_points);

class PSEHead : public MMOCR {
 public:
  explicit PSEHead(const Value& config) : MMOCR(config) {}

  Result<Value> operator()(const Value& _data, const Value& _pred) noexcept {
    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto _preds,
                MakeAvailableOnDevice(_pred["output"].get<Tensor>(), cpu_device, stream_));
    OUTCOME_TRY(stream_.Wait());

    if (_preds.shape().size() != 4 || _preds.shape(0) != 1 ||
        _preds.data_type() != DataType::kFLOAT) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", _preds.shape(),
                     (int)_preds.data_type());
      return Status(eNotSupported);
    }

    // drop batch dimension
    _preds.Squeeze();

    auto channels = static_cast<int>(_preds.shape(0));
    auto height = static_cast<int>(_preds.shape(1));
    auto width = static_cast<int>(_preds.shape(2));

    cv::Mat_<float> preds(_preds.shape(0), height * width, _preds.data<float>());
    sigmoid(preds);

    cv::Mat_<float> score = preds.row(0).reshape(1, height);

    cv::Mat_<uint8_t> masks = preds > min_kernel_confidence_;

    for (int i = 1; i < channels; ++i) {
      masks.row(i) &= masks.row(0);
    }

    cv::Mat_<int32_t> kernel_labels;
    auto region_num = cv::connectedComponents(masks.row(channels - 1).reshape(1, height),
                                              kernel_labels, 4, CV_32S);

    std::vector<int> text_areas;
    std::vector<float> text_scores;
    std::vector<std::vector<int>> text_points;
    contour_expand(masks.rowRange(1, channels), kernel_labels, score, min_kernel_area_, region_num,
                   text_areas, text_scores, text_points);

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

  static void sigmoid(cv::Mat_<float>& score) {
    cv::exp(-score, score);
    score = 1 / (1 + score);
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
};

REGISTER_CODEBASE_COMPONENT(MMOCR, PSEHead);

}  // namespace mmdeploy::mmocr
