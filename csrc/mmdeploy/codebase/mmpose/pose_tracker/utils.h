// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_UTILS_H
#define MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_UTILS_H

#include <array>
#include <numeric>
#include <optional>
#include <vector>

#include "mmdeploy/core/utils/formatter.h"
#include "opencv2/core/core.hpp"
#include "pose_tracker/common.h"

namespace mmdeploy::mmpose::_pose_tracker {

using std::vector;
using Bbox = std::array<float, 4>;
using Bboxes = vector<Bbox>;
using Point = cv::Point2f;
using Points = vector<cv::Point2f>;
using Score = float;
using Scores = vector<float>;

#define POSE_TRACKER_DEBUG(...) MMDEPLOY_DEBUG(__VA_ARGS__)

// opencv3 can't construct cv::Mat from std::array
template <size_t N>
cv::Mat as_mat(const std::array<float, N>& a) {
  return cv::Mat_<float>(a.size(), 1, const_cast<float*>(a.data()));
}

// scale = 1.5, kpt_thr = 0.3
std::optional<Bbox> keypoints_to_bbox(const Points& keypoints, const Scores& scores, float img_h,
                                      float img_w, float scale, float kpt_thr, int min_keypoints);

// xyxy format
float intersection_over_union(const Bbox& a, const Bbox& b);

float object_keypoint_similarity(const Points& pts_a, const Bbox& box_a, const Points& pts_b,
                                 const Bbox& box_b, const vector<float>& sigmas);

template <typename T>
void suppress_non_maximum(const vector<T>& scores, const vector<float>& similarities,
                          vector<int>& is_valid, float thresh);

inline float get_area(const Bbox& bbox) { return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]); }

inline Point get_center(const Bbox& bbox) {
  return {.5f * (bbox[0] + bbox[2]), .5f * (bbox[1] + bbox[3])};
}

inline std::array<float, 2> get_scale(const Bbox& bbox) {
  return {bbox[2] - bbox[0], bbox[3] - bbox[1]};
}

inline Bbox get_bbox(const Point& center, const std::array<float, 2>& scale) {
  return {
      center.x - .5f * scale[0],
      center.y - .5f * scale[1],
      center.x + .5f * scale[0],
      center.y + .5f * scale[1],
  };
}

vector<std::tuple<int, int, float>> greedy_assignment(const vector<float>& scores,
                                                      vector<int>& is_valid_row,
                                                      vector<int>& is_valid_col, float thr);

template <typename T>
inline void suppress_non_maximum(const vector<T>& scores, const vector<float>& similarities,
                                 vector<int>& is_valid, float thresh) {
  assert(is_valid.size() == scores.size());
  vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int i, int j) { return scores[i] > scores[j]; });
  // suppress similar samples
  for (int i = 0; i < indices.size(); ++i) {
    if (auto u = indices[i]; is_valid[u]) {
      for (int j = i + 1; j < indices.size(); ++j) {
        if (auto v = indices[j]; is_valid[v]) {
          if (similarities[u * scores.size() + v] >= thresh) {
            is_valid[v] = false;
          }
        }
      }
    }
  }
}

// TopDownAffine's internal logic for mapping pose model inputs
Bbox map_bbox(const Bbox& box);

}  // namespace mmdeploy::mmpose::_pose_tracker

#endif  // MMDEPLOY_UTILS_H
