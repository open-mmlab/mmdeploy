// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/dbnet.h"

#include <opencv2/imgproc.hpp>

#include "clipper.hpp"
#include "core/device.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "experimental/module_adapter.h"
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
      params_.text_repr_type = params.value("text_repr_type", params_.text_repr_type);
      params_.mask_thr = params.value("mask_thr", params_.mask_thr);
      params_.min_text_score = params.value("min_text_score", params_.min_text_score);
      params_.min_text_width = params.value("min_text_width", params_.min_text_width);
      params_.unclip_ratio = params.value("unclip_ratio", params_.unclip_ratio);
      params_.max_candidates = params.value("max_candidate", params_.max_candidates);
      params_.rescale = params.value("rescale", params_.rescale);
      params_.downsample_ratio = params.value("downsample_ratio", params_.downsample_ratio);
    }
    auto platform_name = Platform(device_.platform_id()).GetPlatformName();
    auto creator = Registry<DbHeadImpl>::Get().GetCreator(platform_name);
    if (!creator) {
      MMDEPLOY_ERROR("DBHead: implementation for platform {} not found", platform_name);
      throw_exception(eEntryNotFound);
    }
    impl_ = creator->Create(nullptr);
    impl_->Init(params_, stream_);
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) const {
    auto conf = _prob["output"].get<Tensor>();
    if (!(conf.shape().size() == 4 && conf.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", conf.shape(),
                     (int)conf.data_type());
      return Status(eNotSupported);
    }

    std::vector<std::vector<cv::Point>> contours;
    std::vector<float> scores;
    OUTCOME_TRY(impl_->Process(conf, contours, scores));

    auto scale_w = 1.f;
    auto scale_h = 1.f;
    if (params_.rescale) {
      scale_w = params_.downsample_ratio / _data["img_metas"]["scale_factor"][0].get<float>();
      scale_h = params_.downsample_ratio / _data["img_metas"]["scale_factor"][1].get<float>();
    }

    TextDetectorOutput output;
    for (int idx = 0; idx < contours.size(); ++idx) {
      auto expanded = unclip(contours[idx], params_.unclip_ratio);
      if (expanded.empty()) {
        continue;
      }
      auto rect = cv::minAreaRect(expanded);
      if ((int)rect.size.width <= params_.min_text_width) {
        continue;
      }
      std::array<cv::Point2f, 4> box_points;
      rect.points(box_points.data());
      auto& bbox = output.boxes.emplace_back();
      for (int i = 0; i < 4; ++i) {
        bbox[i * 2] = box_points[i].x * scale_w;
        bbox[i * 2 + 1] = box_points[i].y * scale_h;
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

  std::unique_ptr<DbHeadImpl> impl_;
  DbHeadParams params_;
};

REGISTER_CODEBASE_COMPONENT(MMOCR, DBHead);

}  // namespace mmocr

MMDEPLOY_DEFINE_REGISTRY(mmocr::DbHeadImpl);

}  // namespace mmdeploy
