// Copyright (c) OpenMMLab. All rights reserved.

#include "smoothing_filter.h"

namespace mmdeploy::mmpose::_pose_tracker {

SmoothingFilter::SmoothingFilter(const Bbox& bbox, const Points& pts,
                                 const SmoothingFilter::Params& params)
    : params_(params),
      pts_v_(pts.size()),
      pts_x_(pts),
      center_v_{},
      center_x_{get_center(bbox)},
      scale_v_{},
      scale_x_{get_scale(bbox)} {}

std::pair<Bbox, Points> SmoothingFilter::Step(const Bbox& bbox, const Points& kpts) {
  constexpr auto abs = [](const Point& p) { return std::sqrt(p.dot(p)); };

  // filter key-points
  step<Point>(pts_x_, pts_v_, kpts, params_, abs);

  // filter bbox center
  std::array c{get_center(bbox)};
  step<Point>(center_x_, center_v_, c, params_, abs);

  // filter bbox scales
  auto s = get_scale(bbox);
  step<float>(scale_x_, scale_v_, s, params_, [](auto x) { return x; });

  return {get_bbox(center_x_[0], scale_x_), pts_x_};
}

void SmoothingFilter::Reset(const Bbox& bbox, const Points& pts) {
  pts_v_ = Points(pts_v_.size());
  center_v_ = {};
  scale_v_ = {};
  pts_x_ = pts;
  center_v_ = {get_center(bbox)};
  scale_v_ = get_scale(bbox);
}

float SmoothingFilter::smoothing_factor(float cutoff) {
  static constexpr float kPi = 3.1415926;
  auto r = 2.f * kPi * cutoff;
  return r / (r + 1.f);
}

}  // namespace mmdeploy::mmpose::_pose_tracker
