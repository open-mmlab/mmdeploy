// Copyright (c) OpenMMLab. All rights reserved.

#include <numeric>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "opencv_utils.h"

namespace mmdeploy {

// warp rotated rect
class WarpBoxes {
 public:
  Result<Value> operator()(const Value& img, const Value& det) {
    Value patches = ValueType::kArray;
    if (det.is_object() && det.contains("boxes")) {
      auto boxes = from_value<std::vector<std::vector<cv::Point>>>(det["boxes"]);
      auto ori_img = mmdeploy::cpu::Mat2CVMat(img["ori_img"].get<mmdeploy::Mat>()).clone();
      for (int i = 0; i < boxes.size(); ++i) {
        auto patch = warp(ori_img, boxes[i]);
        patches.push_back(make_pointer({{"ori_img", cpu::CVMat2Mat(patch, PixelFormat::kBGR)}}));
        //      cv::imwrite(std::to_string(i) + ".png", patch);
      }
    } else {  // whole image as a bbox
      patches.push_back({{"ori_img", img["ori_img"].get<mmdeploy::Mat>()}});
    }
    return patches;
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
    auto m = getAffineTransform(src.data(), dst.data());
    cv::Mat warped;
    warpAffine(img, warped, m, {static_cast<int>(w), static_cast<int>(h)});
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

class WarpBoxesCreator : public Creator<Module> {
 public:
  const char* GetName() const override { return "WarpBoxes"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Module> Create(const Value& value) override { return CreateTask(WarpBoxes{}); }
};

REGISTER_MODULE(Module, WarpBoxesCreator);

}  // namespace mmdeploy
