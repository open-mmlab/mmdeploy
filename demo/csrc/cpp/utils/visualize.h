// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_VISUALIZE_H
#define MMDEPLOY_VISUALIZE_H

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "palette.h"
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
      cv::Mat img;
      if (v.background_ == "black") {
        img = cv::Mat::zeros(frame.size(), CV_8UC3);
      } else {
        img = frame;
        if (img.channels() == 1) {
          cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        }
      }
      if (scale_ != 1) {
        cv::resize(img, img, {}, scale_, scale_);
      } else if (img.data == frame.data) {
        img = img.clone();
      }
      img_ = std::move(img);
    }

    void add_label(int label_id, float score, int index) {
      printf("label: %d, label_id: %d, score: %.4f\n", index, label_id, score);
      auto size = .5f * static_cast<float>(img_.rows + img_.cols);
      offset_ += add_text(to_text(label_id, score), {1, (float)offset_}, size) + 2;
    }

    int add_text(const std::string& text, const cv::Point2f& origin, float size) {
      static constexpr const int font_face = cv::FONT_HERSHEY_SIMPLEX;
      static constexpr const int thickness = 1;
      static constexpr const auto max_font_scale = .5f;
      static constexpr const auto min_font_scale = .25f;
      float font_scale{};
      if (size < 20) {
        font_scale = min_font_scale;
      } else if (size > 200) {
        font_scale = max_font_scale;
      } else {
        font_scale = min_font_scale + (size - 20) / (200 - 20) * (max_font_scale - min_font_scale);
      }
      int baseline{};
      auto text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
      cv::Rect rect(origin + cv::Point2f(0, text_size.height + 2 * thickness),
                    origin + cv::Point2f(text_size.width, 0));
      rect &= cv::Rect({}, img_.size());
      if (rect.area() > 0) {
        img_(rect) *= .35f;
        cv::putText(img_, text, origin + cv::Point2f(0, text_size.height), font_face, font_scale,
                    cv::Scalar::all(255), thickness, cv::LINE_AA);
      }
      return text_size.height;
    }

    static std::string to_text(int label_id, float score) {
      std::stringstream ss;
      ss << label_id << ": " << std::fixed << std::setprecision(1) << score * 100;
      return ss.str();
    }

    template <typename Mask>
    void add_det(const mmdeploy_rect_t& rect, int label_id, float score, const Mask* mask,
                 int index) {
      printf("bbox %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, label=%d, score=%.4f\n", index,
             rect.left, rect.top, rect.right, rect.bottom, label_id, score);
      if (mask) {
        fprintf(stdout, "mask %d, height=%d, width=%d\n", index, mask->height, mask->width);
        auto x0 = (int)std::max(std::floor(rect.left) - 1, 0.f);
        auto y0 = (int)std::max(std::floor(rect.top) - 1, 0.f);
        add_instance_mask({x0, y0}, rand(), mask->data, mask->height, mask->width);
      }
      add_bbox(rect, label_id, score);
    }

    void add_instance_mask(const cv::Point& origin, int color_id, const char* mask_data, int mask_h,
                           int mask_w, float alpha = .5f) {
      auto color = v_.palette_.data[color_id % v_.palette_.data.size()];
      auto x_end = std::min(origin.x + mask_w, img_.cols);
      auto y_end = std::min(origin.y + mask_h, img_.rows);
      auto img_data = img_.ptr<cv::Vec3b>();
      for (int i = origin.y; i < y_end; ++i) {
        for (int j = origin.x; j < x_end; ++j) {
          if (mask_data[(i - origin.y) * mask_w + (j - origin.x)]) {
            img_data[i * img_.cols + j] = img_data[i * img_.cols + j] * (1 - alpha) + color * alpha;
          }
        }
      }
    }

    void add_bbox(mmdeploy_rect_t rect, int label_id, float score) {
      rect.left *= scale_;
      rect.right *= scale_;
      rect.top *= scale_;
      rect.bottom *= scale_;
      if (label_id >= 0 && score > 0) {
        auto area = std::max(0.f, (rect.right - rect.left) * (rect.bottom - rect.top));
        add_text(to_text(label_id, score), {rect.left, rect.top}, std::sqrt(area));
      }
      cv::rectangle(img_, cv::Point2f(rect.left, rect.top), cv::Point2f(rect.right, rect.bottom),
                    cv::Scalar(0, 255, 0));
    }

    void add_text_det(mmdeploy_point_t bbox[4], float score, const char* text, size_t text_size,
                      int index) {
      printf("bbox[%d]: (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), %.2f\n", index,  //
             bbox[0].x, bbox[0].y,                                                               //
             bbox[1].x, bbox[1].y,                                                               //
             bbox[2].x, bbox[2].y,                                                               //
             bbox[3].x, bbox[3].y, score);
      std::vector<cv::Point> poly_points;
      cv::Point2f center{};
      for (int i = 0; i < 4; ++i) {
        poly_points.emplace_back(bbox[i].x * scale_, bbox[i].y * scale_);
        center += cv::Point2f(poly_points.back());
      }
      cv::polylines(img_, poly_points, true, cv::Scalar{0, 255, 0}, 1, cv::LINE_AA);
      if (text) {
        auto area = cv::contourArea(poly_points);
        fprintf(stdout, "text[%d]: %s\n", index, text);
        add_text(std::string(text, text + text_size), center / 4, std::sqrt(area));
      }
    }

    void add_rotated_det(const float bbox[5], int label_id, float score) {
      float xc = bbox[0] * scale_;
      float yc = bbox[1] * scale_;
      float w = bbox[2] * scale_;
      float h = bbox[3] * scale_;
      float ag = bbox[4];
      float wx = w / 2 * std::cos(ag);
      float wy = w / 2 * std::sin(ag);
      float hx = -h / 2 * std::sin(ag);
      float hy = h / 2 * std::cos(ag);
      cv::Point2f p1{xc - wx - hx, yc - wy - hy};
      cv::Point2f p2{xc + wx - hx, yc + wy - hy};
      cv::Point2f p3{xc + wx + hx, yc + wy + hy};
      cv::Point2f p4{xc - wx + hx, yc - wy + hy};
      cv::Point2f c = .25f * (p1 + p2 + p3 + p4);
      cv::drawContours(
          img_,
          std::vector<std::vector<cv::Point>>{{p1 * scale_, p2 * scale_, p3 * scale_, p4 * scale_}},
          -1, {0, 255, 0}, 2, cv::LINE_AA);
      add_text(to_text(label_id, score), c, std::sqrt(w * h));
    }

    void add_mask(int height, int width, int n_classes, const int* mask, const float* score) {
      cv::Mat color_mask = cv::Mat::zeros(height, width, CV_8UC3);
      auto n_pix = color_mask.total();

      // compute top 1 idx if score (CHW) is available
      cv::Mat_<int> top;
      if (!mask && score) {
        top = cv::Mat_<int>::zeros(height, width);
        for (auto c = 1; c < n_classes; ++c) {
          top.forEach([&](int& x, const int* idx) {
            auto offset = idx[0] * width + idx[1];
            if (score[c * n_pix + offset] > score[x * n_pix + offset]) {
              x = c;
            }
          });
        }
        mask = top.ptr<int>();
      }

      if (mask) {
        // palette look-up
        color_mask.forEach<cv::Vec3b>([&](cv::Vec3b& x, const int* idx) {
          auto& palette = v_.palette_.data;
          x = palette[mask[idx[0] * width + idx[1]] % palette.size()];
        });

        if (color_mask.size() != img_.size()) {
          cv::resize(color_mask, color_mask, img_.size());
        }

        // blend mask and background image
        cv::addWeighted(img_, .5, color_mask, .5, 0., img_);
      }
    }

    void add_pose(const mmdeploy_point_t* pts, const float* scores, int32_t pts_size, double thr) {
      auto& skel = v_.skeleton_;
      if (skel.point_colors.size() != pts_size) {
        std::cout << "error: mismatched number of keypoints: " << skel.point_colors.size() << " vs "
                  << pts_size << ", skip pose visualization.\n";
        return;
      }
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
    int offset_{1};
    cv::Mat img_;
  };

  explicit Visualize(int size = 0) : size_(size) { palette_ = Palette::get(32); }

  Session get_session(const cv::Mat& frame) { return Session(*this, frame); }

  void set_skeleton(const Skeleton& skeleton) { skeleton_ = skeleton; }

  void set_palette(const Palette& palette) { palette_ = palette; }

  void set_background(const std::string& background) { background_ = background; }

 private:
  friend Session;
  Skeleton skeleton_;
  Palette palette_;
  std::string background_;
  int size_{};
};

}  // namespace utils

#endif  // MMDEPLOY_VISUALIZE_H
