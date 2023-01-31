// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_VISUALIZE_H
#define MMDEPLOY_VISUALIZE_H

#include <iomanip>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "skeleton.h"

namespace utils {

class Visualize {
 public:
  class Session {
   public:
    explicit Session(Visualize& v, const cv::Mat& frame) : v_(v) {
      if (v_.size_) {
        scale_ = (float)v_.size_ / (float)std::max(frame.cols, frame.rows);
      }
      if (scale_ != 1) {
        cv::resize(frame, img_, {}, scale_, scale_);
      } else {
        img_ = frame.clone();
      }
    }

    void add_label(int label_id, float score) {
      static constexpr const int font_face = cv::FONT_HERSHEY_SIMPLEX;
      static constexpr const int thickness = 1;
      static constexpr const double font_scale = 0.5;
      int baseline = 0;
      std::stringstream ss;
      ss << std::to_string(label_id) << ": " << std::setw(6) << std::fixed << score * 100;
      auto text_size = cv::getTextSize(ss.str(), font_face, font_scale, thickness, &baseline);
      offset_ += text_size.height;
      cv::Point origin(0, offset_);
      cv::rectangle(img_, origin + cv::Point(0, thickness),
                    origin + cv::Point(text_size.width, -text_size.height), cv::Scalar{},
                    cv::FILLED);
      cv::putText(img_, ss.str(), origin, font_face, font_scale, cv::Scalar::all(255), thickness,
                  cv::LINE_8);
      offset_ += thickness;
    }

    // TODO: show label & score
    template <typename Mask>
    void add_det(const mmdeploy_rect_t& rect, int label_id, float score, const Mask* mask) {
      add_bbox(rect, label_id, score);
      if (mask) {
        cv::Mat mask_img(mask->height, mask->width, CV_8UC1, const_cast<char*>(mask->data));
        auto x0 = std::max(std::floor(rect.left) - 1, 0.f);
        auto y0 = std::max(std::floor(rect.top) - 1, 0.f);
        cv::Rect roi((int)x0, (int)y0, mask->width, mask->height);

        // split the RGB channels, overlay mask to a specific color channel
        cv::Mat ch[3]{};
        cv::split(img_, ch);
        int col = 0;  // int col = i % 3;
        cv::bitwise_or(mask_img, ch[col](roi), ch[col](roi));
        merge(ch, 3, img_);
      }
    }

    // TODO: show label & score
    void add_bbox(const mmdeploy_rect_t& rect, int label_id, float score) {
      cv::rectangle(img_, cv::Point2f(rect.left, rect.top), cv::Point2f(rect.right, rect.bottom),
                    cv::Scalar(0, 255, 0));
    }

    // TODO: show text
    void add_text_det(mmdeploy_point_t bbox[4], float score, const char* text, size_t text_size) {
      std::vector<cv::Point2f> poly_points;
      for (int i = 0; i < 4; ++i) {
        poly_points.emplace_back(bbox[i].x * scale_, bbox[i].y * scale_);
      }
      cv::polylines(img_, poly_points, true, cv::Scalar{0, 255, 0});
    }

    void add_rotated_det(const float bbox[5], int label_id, float score) {
      float xc = bbox[0];
      float yc = bbox[1];
      float w = bbox[2];
      float h = bbox[3];
      float ag = bbox[4];
      float wx = w / 2 * std::cos(ag);
      float wy = w / 2 * std::sin(ag);
      float hx = -h / 2 * std::sin(ag);
      float hy = h / 2 * std::cos(ag);
      cv::Point2f p1{xc - wx - hx, yc - wy - hy};
      cv::Point2f p2{xc + wx - hx, yc + wy - hy};
      cv::Point2f p3{xc + wx + hx, yc + wy + hy};
      cv::Point2f p4{xc - wx + hx, yc - wy + hy};
      cv::drawContours(
          img_,
          std::vector<std::vector<cv::Point>>{{p1 * scale_, p2 * scale_, p3 * scale_, p4 * scale_}},
          -1, {0, 255, 0}, 2);
    }

    // TODO: handle score output
    void add_mask(int height, int width, int n_classes, const int* mask, const float* score) {
      cv::Mat color_mask = cv::Mat::zeros(height, width, CV_8UC3);
      int pos = 0;
      for (auto iter = color_mask.begin<cv::Vec3b>(); iter != color_mask.end<cv::Vec3b>(); ++iter) {
        *iter = v_.palette_[mask[pos++] % v_.palette_.size()];
      }
      if (color_mask.size() != img_.size()) {
        cv::resize(color_mask, color_mask, img_.size(), 0., 0.);
      }
      img_ = img_ * 0.5 + color_mask * 0.5;
    }

    void add_pose(const mmdeploy_point_t* pts, const float* scores, int32_t pts_size, double thr) {
      auto& skel = v_.skel_;
      std::vector<int> used(pts_size);
      std::vector<int> is_end_point(pts_size);
      for (size_t i = 0; i < skel.links.size(); ++i) {
        auto u = skel.links[i].first;
        auto v = skel.links[i].second;
        is_end_point[u] = is_end_point[v] = 1;
        if (scores[u] > thr && scores[v] > thr) {
          used[u] = used[v] = 1;
          cv::Point2f p0(pts[u].x, pts[u].y);
          cv::Point2f p1(pts[v].x, pts[v].y);
          cv::line(img_, p0 * scale_, p1 * scale_, skel.palette[skel.link_colors[i]], 1,
                   cv::LINE_AA);
        }
      }
      for (size_t i = 0; i < pts_size; ++i) {
        if (!is_end_point[i] && scores[i] > thr || used[i]) {
          cv::Point2f p(pts[i].x, pts[i].y);
          cv::circle(img_, p * scale_, 1, skel.palette[skel.point_colors[i]], 2, cv::LINE_AA);
        }
      }
    }

    cv::Mat get() { return img_; }

   private:
    Visualize& v_;
    float scale_{1};
    int offset_{};
    cv::Mat img_;
  };

  explicit Visualize(int size) : size_(size) {}

  Session get_session(const cv::Mat& frame) { return Session(*this, frame); }

  void set_skeleton(const Skeleton& skel) { skel_ = skel; }

  void set_palette(const std::vector<cv::Vec3b>& palette) { palette_ = palette; }

 private:
  friend Session;
  Skeleton skel_;
  std::vector<cv::Vec3b> palette_;
  int size_{};
};

}  // namespace utils

#endif  // MMDEPLOY_VISUALIZE_H
