// Copyright (c) OpenMMLab. All rights reserved.

#include <numeric>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv_utils.h"

namespace mmdeploy::mmocr {

// Warp rotated rect
class WarpBbox {
 public:
  Result<Value> operator()(const Value& img, const Value& det) {
    auto ori_img = img["ori_img"].get<framework::Mat>();
    if (det.is_object() && det.contains("bbox")) {
      auto bbox = from_value<std::vector<cv::Point>>(det["bbox"]);
      auto patch = warp(mmdeploy::cpu::Mat2CVMat(ori_img), bbox);
      return Value{{"ori_img", cpu::CVMat2Mat(patch, ori_img.pixel_format())}};
    } else {  // whole image as a bbox
      return Value{{"ori_img", ori_img}};
    }
  }

  // assuming rect
  static cv::Mat warp(const cv::Mat& img, const std::vector<cv::Point>& _pts) {
    auto pts = sort_vertex(_pts);
    std::vector<cv::Point2f> src(begin(pts), end(pts));
    auto e0 = norm(pts[0] - pts[1]);
    auto e1 = norm(pts[1] - pts[2]);
    auto w = static_cast<float>(std::max(e0, e1));
    auto h = static_cast<float>(std::min(e0, e1));
    std::vector<cv::Point2f> dst{{0, 0}, {w, 0}, {w, h}, {0, h}};
    auto m = cv::getAffineTransform(src.data(), dst.data());
    cv::Mat warped;
    cv::warpAffine(img, warped, m, {static_cast<int>(w), static_cast<int>(h)});
    return warped;
  }

  static std::vector<cv::Point> sort_vertex(std::vector<cv::Point> ps) {
    auto pivot = *min_element(begin(ps), end(ps), [](auto r, auto p) {
      return (r.y != p.y) ? (r.y < p.y) : (r.x < p.x);
    });
    // TODO: resolve tie with distance
    sort(begin(ps), end(ps), [&](auto a, auto b) {
      if (a == pivot) return b != pivot;
      return (a - pivot).cross(b - pivot) > 0;
    });
    auto tl = accumulate(begin(ps) + 1, end(ps), ps[0], [](auto r, auto p) {
      return cv::Point{std::min(r.x, p.x), std::min(r.y, p.y)};
    });
    auto cmp = [&](auto r, auto p) {
      cv::Point2f u{r - tl}, v{p - tl};
      return u.dot(u) < v.dot(v);
    };
    auto tl_idx = min_element(begin(ps), end(ps), cmp) - begin(ps);
    rotate(begin(ps), begin(ps) + tl_idx, end(ps));
    return ps;
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (WarpBbox, 0),
                               [](const Value&) { return CreateTask(WarpBbox{}); });

}  // namespace mmdeploy::mmocr
