// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_SMOOTHING_FILTER_H
#define MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_SMOOTHING_FILTER_H

#include "mmdeploy/core/mpl/span.h"
#include "pose_tracker/utils.h"

namespace mmdeploy::mmpose::_pose_tracker {

template <typename T>
using span = mmdeploy::Span<T>;

class SmoothingFilter {
 public:
  struct Params {
    float beta;
    float fc_min;
    float fc_v;
  };
  explicit SmoothingFilter(const Bbox& bbox, const Points& pts, const Params& params);

  std::pair<Bbox, Points> Step(const Bbox& bbox, const Points& kpts);

  void Reset(const Bbox& bbox, const Points& pts);

 private:
  static float smoothing_factor(float cutoff);

  template <typename T, typename Norm>
  static void step(span<T> x, span<T> v, span<const T> x1, const Params& params, Norm norm) {
    auto a_v = smoothing_factor(params.fc_v);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = smooth(a_v, v[i], x1[i] - x[i]);
      auto fc = params.fc_min + params.beta * norm(v[i]);
      auto a_x = smoothing_factor(fc);
      x[i] = smooth(a_x, x[i], x1[i]);
    }
  }

  template <typename T>
  static T smooth(float a, const T& x0, const T& x1) {
    return (1.f - a) * x0 + a * x1;
  }

 private:
  Params params_;
  std::vector<Point> pts_v_;
  std::vector<Point> pts_x_;
  std::array<Point, 1> center_v_;
  std::array<Point, 1> center_x_;
  std::array<float, 2> scale_v_;
  std::array<float, 2> scale_x_;
};

}  // namespace mmdeploy::mmpose::_pose_tracker

#endif  // MMDEPLOY_SMOOTHING_FILTER_H
