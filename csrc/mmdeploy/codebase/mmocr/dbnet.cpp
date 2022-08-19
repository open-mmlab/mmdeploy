// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/dbnet.h"

#include <opencv2/imgproc.hpp>

#include "clipper.hpp"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmocr.h"

namespace mmdeploy {

namespace mmocr {

using std::string;
using std::vector;

class DBHead : public MMOCR {
 public:
  explicit DBHead(const Value& config) : MMOCR(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      text_repr_type_ = params.value("text_repr_type", text_repr_type_);
      mask_thr_ = params.value("mask_thr", mask_thr_);
      min_text_score_ = params.value("min_text_score", min_text_score_);
      min_text_width_ = params.value("min_text_width", min_text_width_);
      unclip_ratio_ = params.value("unclip_ratio", unclip_ratio_);
      max_candidates_ = params.value("max_candidate", max_candidates_);
      rescale_ = params.value("rescale", rescale_);
      downsample_ratio_ = params.value("downsample_ratio", downsample_ratio_);
    }
    auto platform = Platform(device_.platform_id()).GetPlatformName();
    auto creator = Registry<DbHeadImpl>::Get().GetCreator(platform);
    if (!creator) {
      MMDEPLOY_ERROR(
          "DBHead: implementation for platform \"{}\" not found. Available platforms: {}", platform,
          Registry<DbHeadImpl>::Get().List());
      throw_exception(eEntryNotFound);
    }
    impl_ = creator->Create(nullptr);
    impl_->Init(stream_);
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) const {
    auto conf = _prob["output"].get<Tensor>();
    if (!(conf.shape().size() == 4 && conf.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", conf.shape(),
                     (int)conf.data_type());
      return Status(eNotSupported);
    }

    // drop batch dimension
    conf.Squeeze(0);

    conf = conf.Slice(0);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<float> scores;
    OUTCOME_TRY(impl_->Process(conf, mask_thr_, max_candidates_, contours, scores));

    auto scale_w = 1.f;
    auto scale_h = 1.f;
    if (rescale_) {
      scale_w /= downsample_ratio_ * _data["img_metas"]["scale_factor"][0].get<float>();
      scale_h /= downsample_ratio_ * _data["img_metas"]["scale_factor"][1].get<float>();
    }

    TextDetectorOutput output;
    for (int idx = 0; idx < contours.size(); ++idx) {
      if (scores[idx] < min_text_score_) {
        continue;
      }
      auto expanded = unclip(contours[idx], unclip_ratio_);
      if (expanded.empty()) {
        continue;
      }
      auto rect = cv::minAreaRect(expanded);
      if ((int)rect.size.width <= min_text_width_) {
        continue;
      }
      std::array<cv::Point2f, 4> box_points;
      rect.points(box_points.data());
      auto& bbox = output.boxes.emplace_back();
      for (int i = 0; i < 4; ++i) {
        // ! performance metrics drops without rounding here
        bbox[i * 2] = cvRound(box_points[i].x * scale_w);
        bbox[i * 2 + 1] = cvRound(box_points[i].y * scale_h);
      }
      output.scores.push_back(scores[idx]);
    }

    return to_value(output);
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

  std::string text_repr_type_{"quad"};
  float mask_thr_{.3};
  float min_text_score_{.3};
  int min_text_width_{5};
  float unclip_ratio_{1.5};
  int max_candidates_{3000};
  bool rescale_{true};
  float downsample_ratio_{1.};

  std::unique_ptr<DbHeadImpl> impl_;
};

REGISTER_CODEBASE_COMPONENT(MMOCR, DBHead);

}  // namespace mmocr

MMDEPLOY_DEFINE_REGISTRY(mmocr::DbHeadImpl);

}  // namespace mmdeploy
