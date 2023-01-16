// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_tracker/utils.h"

namespace mmdeploy::mmpose::_pose_tracker {

vector<std::tuple<int, int, float>> greedy_assignment(const vector<float>& scores,
                                                      vector<int>& is_valid_row,
                                                      vector<int>& is_valid_col, float thr) {
  const auto n_rows = is_valid_row.size();
  const auto n_cols = is_valid_col.size();
  vector<std::tuple<int, int, float>> assignment;
  assignment.reserve(std::max(n_rows, n_cols));
  while (true) {
    auto max_score = std::numeric_limits<float>::lowest();
    int max_row = -1;
    int max_col = -1;
    for (int i = 0; i < n_rows; ++i) {
      if (is_valid_row[i]) {
        for (int j = 0; j < n_cols; ++j) {
          if (is_valid_col[j]) {
            if (scores[i * n_cols + j] > max_score) {
              max_score = scores[i * n_cols + j];
              max_row = i;
              max_col = j;
            }
          }
        }
      }
    }
    if (max_score < thr) {
      break;
    }
    is_valid_row[max_row] = 0;
    is_valid_col[max_col] = 0;
    assignment.emplace_back(max_row, max_col, max_score);
  }
  return assignment;
}

float intersection_over_union(const Bbox& a, const Bbox& b) {
  auto x1 = std::max(a[0], b[0]);
  auto y1 = std::max(a[1], b[1]);
  auto x2 = std::min(a[2], b[2]);
  auto y2 = std::min(a[3], b[3]);

  auto inter_area = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);

  auto a_area = get_area(a);
  auto b_area = get_area(b);
  auto union_area = a_area + b_area - inter_area;

  if (union_area == 0.f) {
    return 0;
  }

  return inter_area / union_area;
}

float object_keypoint_similarity(const Points& pts_a, const Bbox& box_a, const Points& pts_b,
                                 const Bbox& box_b, const vector<float>& sigmas) {
  assert(pts_a.size() == sigmas.size());
  assert(pts_b.size() == sigmas.size());
  auto scale = [](const Bbox& bbox) -> float {
    auto a = bbox[2] - bbox[0];
    auto b = bbox[3] - bbox[1];
    return std::sqrt(a * a + b * b);
  };
  auto oks = [](const Point& pa, const Point& pb, float s, float k) {
    return std::exp(-(pa - pb).dot(pa - pb) / (2.f * s * s * k * k));
  };
  auto sum = 0.f;
  const auto s = .5f * (scale(box_a) + scale(box_b));
  for (int i = 0; i < sigmas.size(); ++i) {
    sum += oks(pts_a[i], pts_b[i], s, sigmas[i]);
  }
  sum /= static_cast<float>(sigmas.size());
  return sum;
}

std::optional<Bbox> keypoints_to_bbox(const Points& keypoints, const Scores& scores, float img_h,
                                      float img_w, float scale, float kpt_thr, int min_keypoints) {
  int valid = 0;
  auto x1 = static_cast<float>(img_w);
  auto y1 = static_cast<float>(img_h);
  auto x2 = 0.f;
  auto y2 = 0.f;
  for (size_t i = 0; i < keypoints.size(); ++i) {
    auto& kpt = keypoints[i];
    if (scores[i] >= kpt_thr) {
      x1 = std::min(x1, kpt.x);
      y1 = std::min(y1, kpt.y);
      x2 = std::max(x2, kpt.x);
      y2 = std::max(y2, kpt.y);
      ++valid;
    }
  }
  if (min_keypoints < 0) {
    min_keypoints = (static_cast<int>(scores.size()) + 1) / 2;
  }
  if (valid < min_keypoints) {
    return std::nullopt;
  }
  auto xc = .5f * (x1 + x2);
  auto yc = .5f * (y1 + y2);
  auto w = (x2 - x1) * scale;
  auto h = (y2 - y1) * scale;

  return std::array<float, 4>{
      std::max(0.f, std::min(img_w, xc - .5f * w)),
      std::max(0.f, std::min(img_h, yc - .5f * h)),
      std::max(0.f, std::min(img_w, xc + .5f * w)),
      std::max(0.f, std::min(img_h, yc + .5f * h)),
  };
}

Bbox map_bbox(const Bbox& box) {
  Point p0(box[0], box[1]);
  Point p1(box[2], box[3]);
  auto c = .5f * (p0 + p1);
  auto s = p1 - p0;
  static constexpr std::array image_size{192.f, 256.f};
  float aspect_ratio = image_size[0] * 1.0 / image_size[1];
  if (s.x > aspect_ratio * s.y) {
    s.y = s.x / aspect_ratio;
  } else if (s.x < aspect_ratio * s.y) {
    s.x = s.y * aspect_ratio;
  }
  s.x *= 1.25f;
  s.y *= 1.25f;
  p0 = c - .5f * s;
  p1 = c + .5f * s;
  return {p0.x, p0.y, p1.x, p1.y};
}

}  // namespace mmdeploy::mmpose::_pose_tracker
