// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "codebase/mmocr/mmocr.h"
#include "core/device.h"
#include "core/registry.h"
#include "core/serialization.h"
#include "core/utils/device_utils.h"

namespace mmdeploy::mmocr {

std::vector<std::vector<float>> pixel_group_cpu(const cv::Mat_<float>& score,
                                                const cv::Mat_<uint8_t>& mask,
                                                const Tensor& embedding,
                                                const cv::Mat_<int32_t>& kernel_label,
                                                const cv::Mat_<uint8_t>& kernel_contour,
                                                int kernel_region_num, float dis_threshold);

class PANHead : public MMOCR {
 public:
  explicit PANHead(const Value& config) : MMOCR(config) {}

  Result<Value> operator()(const Value& _data, const Value& _pred) {
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

    auto _score = pred.Slice(1);
    cv::Mat_<float> score(_score.shape(1), _score.shape(2), _score.data<float>());
    sigmoid(score);

    auto _embed = pred.Slice(2, -1);
    cv::Mat_<float> embed(_embed.shape(0), _embed.shape(1) * _embed.shape(2),
                          _embed.data<float>());  // C x HW
    cv::transpose(embed, embed);                  // HW x C
    _embed.Reshape({_embed.shape(1), _embed.shape(2), _embed.shape(0)});

    cv::Mat_<uint8_t> kernel = score > min_kernel_confidence_;

    cv::Mat_<int32_t> labels;
    auto region_num = cv::connectedComponents(kernel, labels, 4, CV_32S);

    cv::Mat contours;
    cv::findContours(kernel, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::Mat_<uchar> kernel_contours;
    cv::drawContours(kernel_contours, contours, -1, 255);

    auto text_points = pixel_group_cpu(score, kernel, _embed, labels, kernel_contours, region_num,
                                       min_text_avg_confidence_);

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
        auto scale_w = _data["img_metas"]["scale_factor"][0].get<float>();
        auto scale_h = _data["img_metas"]["scale_factor"][1].get<float>();
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
  float min_text_area_;
  bool rescale_{true};
  float downsample_ratio_{1.};
};

REGISTER_CODEBASE_COMPONENT(MMOCR, PANHead);

}  // namespace mmdeploy::mmocr
