// Copyright (c) OpenMMLab. All rights reserved.

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "clipper.hpp"
#include "core/device.h"
#include "core/registry.h"
#include "core/serialization.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "core/value.h"
#include "experimental/module_adapter.h"
#include "mmocr.h"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform_utils.h"

namespace mmdeploy::mmocr {

using std::string;
using std::vector;

class DBHead : public MMOCR {
 public:
  explicit DBHead(const Value& config) : MMOCR(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      text_repr_type_ = params.value("text_repr_type", string{"quad"});
      mask_thr_ = params.value("mask_thr", 0.3f);
      min_text_score_ = params.value("min_text_score", 0.3f);
      min_text_width_ = params.value("min_text_width", 5);
      unclip_ratio_ = params.value("unclip_ratio", 1.5f);
      max_candidates_ = params.value("max_candidate", 3000);
      rescale_ = params.value("rescale", true);
      downsample_ratio_ = params.value("downsample_ratio", 1.0f);
    }
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) {
    DEBUG("preprocess_result: {}", _data);
    DEBUG("inference_result: {}", _prob);

    auto img = _data["img"].get<Tensor>();
    DEBUG("img shape: {}", img.shape());

    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto conf,
                MakeAvailableOnDevice(_prob["output"].get<Tensor>(), cpu_device, stream_));
    OUTCOME_TRY(stream_.Wait());
    DEBUG("shape: {}", conf.shape());

    auto h = conf.shape(2);
    auto w = conf.shape(3);
    auto data = conf.buffer().GetNative();

    cv::Mat score_map((int)h, (int)w, CV_32F, data);

    //    cv::imwrite("conf.png", score_map * 255.);

    cv::Mat text_mask;
    cv::threshold(score_map, text_mask, mask_thr_, 1.f, cv::THRESH_BINARY);

    text_mask.convertTo(text_mask, CV_8U, 255);
    //    cv::imwrite("text_mask.png", text_mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(text_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    if (contours.size() > max_candidates_) {
      contours.resize(max_candidates_);
    }

    TextDetectorOutput output;
    for (auto& poly : contours) {
      auto epsilon = 0.01 * cv::arcLength(poly, true);
      std::vector<cv::Point> approx;
      cv::approxPolyDP(poly, approx, epsilon, true);
      if (approx.size() < 4) {
        continue;
      }
      auto score = box_score_fast(score_map, approx);
      if (score < min_text_score_) {
        continue;
      }
      approx = unclip(approx, unclip_ratio_);
      if (approx.empty()) {
        continue;
      }

      if (text_repr_type_ == "quad") {
        auto rect = cv::minAreaRect(approx);
        if ((int)rect.size.width <= min_text_width_) continue;
        std::vector<cv::Point2f> box_points(4);
        rect.points(box_points.data());
        approx.assign(begin(box_points), end(box_points));
      } else if (text_repr_type_ == "poly") {
      } else {
        assert(0);
      }
      DEBUG("score: {}", score);
      //      cv::drawContours(score_map, vector<vector<cv::Point>>{approx}, -1, 1);

      vector<cv::Point2f> scaled(begin(approx), end(approx));

      if (rescale_) {
        auto scale_w = _data["img_metas"]["scale_factor"][0].get<float>();
        auto scale_h = _data["img_metas"]["scale_factor"][1].get<float>();
        for (auto& p : scaled) {
          p.x /= scale_w * downsample_ratio_;
          p.y /= scale_h * downsample_ratio_;
        }
      }

      auto& bbox = output.boxes.emplace_back();
      for (int i = 0; i < 4; ++i) {
        bbox[i * 2] = scaled[i].x;
        bbox[i * 2 + 1] = scaled[i].y;
      }
      output.scores.push_back(score);
    }

    return to_value(output);
  }

  static float box_score_fast(const cv::Mat& bitmap, const std::vector<cv::Point>& box) noexcept {
    auto rect = cv::boundingRect(box) & cv::Rect({}, bitmap.size());

    cv::Mat mask(rect.size(), CV_8U, cv::Scalar(0));

    cv::fillPoly(mask, std::vector{box}, 1, cv::LINE_8, 0, -rect.tl());
    auto mean = cv::mean(bitmap(rect), mask)[0];
    return static_cast<float>(mean);
  }

  static std::vector<cv::Point> unclip(std::vector<cv::Point>& box, float unclip_ratio) {
    namespace cl = ClipperLib;

    auto area = cv::contourArea(box);
    auto length = cv::arcLength(box, true);
    auto distance = area * unclip_ratio / length;

    cl::Path src;
    transform(begin(box), end(box), back_inserter(src), [](auto p) {
      return cl::IntPoint{p.x, p.y};
    });

    cl::ClipperOffset offset;
    offset.AddPath(src, cl::jtRound, cl::etClosedPolygon);

    std::vector<cl::Path> dst;
    offset.Execute(dst, distance);
    if (dst.size() != 1) {
      return {};
    }

    std::vector<cv::Point> ret;
    transform(begin(dst[0]), end(dst[0]), back_inserter(ret), [](auto p) {
      return cv::Point{static_cast<int>(p.X), static_cast<int>(p.Y)};
    });
    return ret;
  }

 private:
  std::string text_repr_type_{"quad"};
  float mask_thr_{.3};
  float min_text_score_{.3};
  int min_text_width_{5};
  float unclip_ratio_{1.5};
  int max_candidates_{3000};
  bool rescale_{true};
  float downsample_ratio_{1.};
};

REGISTER_CODEBASE_COMPONENT(MMOCR, DBHead);

}  // namespace mmdeploy::mmocr
