// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <opencv2/opencv.hpp>

#include "codebase/mmocr/mmocr.h"
#include "core/device.h"
#include "core/registry.h"
#include "core/serialization.h"
#include "core/utils/device_utils.h"

namespace mmdeploy::mmocr {

std::vector<std::vector<float>> pixel_group_cpu(const cv::Mat_<float>& score,
                                                const cv::Mat_<uint8_t>& mask,
                                                const cv::Mat_<float>& embedding,
                                                const cv::Mat_<int32_t>& kernel_label,
                                                const cv::Mat_<uint8_t>& kernel_contour,
                                                int kernel_region_num, float dis_threshold);

class PANHead : public MMOCR {
 public:
  explicit PANHead(const Value& config) : MMOCR(config) {}

  Result<Value> operator()(const Value& _data, const Value& _pred) noexcept {
    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto pred,
                MakeAvailableOnDevice(_pred["output"].get<Tensor>(), cpu_device, stream_));
    OUTCOME_TRY(stream_.Wait());

    if (pred.shape().size() != 4 || pred.shape(0) != 1 || pred.data_type() != DataType::kFLOAT) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", pred.shape(),
                     (int)pred.data_type());
      return Status(eNotSupported);
    }

    pred.Squeeze();

    auto _text_score = pred.Slice(0);
    cv::Mat_<float> text_score(_text_score.shape(1), _text_score.shape(2),
                               _text_score.data<float>());
    sigmoid(text_score);

    cv::Mat_<uint8_t> text = text_score > min_text_confidence_;

    auto _kernel_score = pred.Slice(1);
    cv::Mat_<float> kernel_score(_kernel_score.shape(1), _kernel_score.shape(2),
                                 _kernel_score.data<float>());
    sigmoid(kernel_score);

    cv::Mat_<uint8_t> kernel = (kernel_score > min_kernel_confidence_) & text;

    auto _embed = pred.Slice(2, pred.shape(0));
    cv::Mat_<float> embed(_embed.shape(0), _embed.shape(1) * _embed.shape(2),
                          _embed.data<float>());  // C x HW
    cv::transpose(embed, embed);                  // HW x C

    cv::Mat_<int32_t> labels;
    auto region_num = cv::connectedComponents(kernel, labels, 4, CV_32S);

    auto text_points = pixel_group_cpu(text_score, text, embed, labels, kernel, region_num,
                                       min_text_avg_confidence_);

    auto scale_w = _data["img_metas"]["scale_factor"][0].get<float>();
    auto scale_h = _data["img_metas"]["scale_factor"][1].get<float>();

    TextDetectorOutput output;
    for (int text_index = 0; text_index != text_points.size(); ++text_index) {
      auto& text_point = text_points[text_index];
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

  static void sigmoid(cv::Mat_<float>& score) {
    cv::exp(-score, score);
    score = 1 / (1 + score);
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
};

REGISTER_CODEBASE_COMPONENT(MMOCR, PANHead);

}  // namespace mmdeploy::mmocr
