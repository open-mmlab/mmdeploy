// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_tracker/tracking_filter.h"

namespace mmdeploy::mmpose::_pose_tracker {

float get_mean_scale(float scale_w, float scale_h) { return std::sqrt(scale_w * scale_h); }

TrackingFilter::TrackingFilter(const Bbox& bbox, const vector<Point>& kpts,
                               float std_weight_position, float std_weight_velocity)
    : std_weight_position_(std_weight_position), std_weight_velocity_(std_weight_velocity) {
  auto center = get_center(bbox);
  auto scale = get_scale(bbox);

  auto mean_scale = get_mean_scale(scale[0], scale[1]);

  const auto n = kpts.size();
  pt_filters_.resize(n);
  for (int i = 0; i < n; ++i) {
    auto& f = pt_filters_[i];
    f.init(4, 2);
    SetKeyPointTransitionMat(i);
    SetKeyPointMeasurementMat(i);

    ResetKeyPoint(i, kpts[i], mean_scale);
  }

  {
    // [x, y, w, h, dx, dy, dw, dh]
    auto& f = bbox_filter_;

    f.init(8, 4);

    SetBboxTransitionMat();
    SetBboxMeasurementMat();

    SetBboxErrorCov(2 * std_weight_position * mean_scale,  //
                    10 * std_weight_velocity * mean_scale);

    f.statePost.at<float>(0) = center.x;
    f.statePost.at<float>(1) = center.y;
    f.statePost.at<float>(2) = scale[0];
    f.statePost.at<float>(3) = scale[1];
  }
}

std::pair<Bbox, Points> TrackingFilter::Predict() {
  auto mean_scale = get_mean_scale(bbox_filter_.statePost.at<float>(2),  //
                                   bbox_filter_.statePost.at<float>(3));
  const auto n = pt_filters_.size();
  Points pts(n);
  for (int i = 0; i < n; ++i) {
    SetKeyPointProcessCov(i, std_weight_position_ * mean_scale, std_weight_velocity_ * mean_scale);
    auto mat = pt_filters_[i].predict();
    pts[i].x = mat.at<float>(0);
    pts[i].y = mat.at<float>(1);
  }
  Bbox bbox;
  {
    SetBboxProcessCov(std_weight_position_ * mean_scale, std_weight_velocity_ * mean_scale);
    auto mat = bbox_filter_.predict();
    auto x = mat.ptr<float>();
    bbox = get_bbox({x[0], x[1]}, {x[2], x[3]});
  }
  return {bbox, pts};
}

std::pair<Bbox, Points> TrackingFilter::Correct(const Bbox& bbox, const Points& kpts,
                                                const vector<bool>& tracked) {
  auto mean_scale = get_mean_scale(bbox_filter_.statePre.at<float>(2),  //
                                   bbox_filter_.statePre.at<float>(3));
  const auto n = pt_filters_.size();
  Points corr_kpts(n);
  for (int i = 0; i < n; ++i) {
    if (!tracked.empty() && tracked[i]) {
      SetKeyPointMeasurementCov(i, std_weight_position_ * mean_scale);
      std::array<float, 2> m{kpts[i].x, kpts[i].y};
      auto mat = pt_filters_[i].correct(as_mat(m));
      corr_kpts[i].x = mat.at<float>(0);
      corr_kpts[i].y = mat.at<float>(1);
    } else {
      ResetKeyPoint(i, kpts[i], mean_scale);
      corr_kpts[i] = kpts[i];
    }
  }
  Bbox corr_bbox;
  {
    SetBboxMeasurementCov(std_weight_position_ * mean_scale);
    auto c = get_center(bbox);
    auto s = get_scale(bbox);
    std::array<float, 4> m{c.x, c.y, s[0], s[1]};
    auto mat = bbox_filter_.correct(as_mat(m));
    auto x = mat.ptr<float>();
    corr_bbox = get_bbox({x[0], x[1]}, {x[2], x[3]});
  }
  return {corr_bbox, corr_kpts};
}

float TrackingFilter::BboxDistance(const Bbox& bbox) {
  auto mean_scale = get_mean_scale(bbox_filter_.statePre.at<float>(2),  //
                                   bbox_filter_.statePre.at<float>(3));
  SetBboxMeasurementCov(std_weight_position_ * mean_scale);
  auto c = get_center(bbox);
  auto s = get_scale(bbox);
  std::array<float, 4> m{c.x, c.y, s[0], s[1]};
  cv::Mat z = as_mat(m);
  auto& f = bbox_filter_;
  cv::Mat sigma;
  cv::gemm(f.measurementMatrix * f.errorCovPre, f.measurementMatrix, 1, f.measurementNoiseCov, 1,
           sigma, cv::GEMM_2_T);
  cv::Mat r = z - f.measurementMatrix * f.statePre;
  // ignore contribution of scales as it is unstable when inferred from key-points
  r.at<float>(2) = 0;
  r.at<float>(3) = 0;
  cv::Mat d = r.t() * sigma.inv() * r;
  return d.at<float>();
}

vector<float> TrackingFilter::KeyPointDistance(const Points& kpts) {
  auto mean_scale = get_mean_scale(bbox_filter_.statePre.at<float>(2),  //
                                   bbox_filter_.statePre.at<float>(3));

  const auto n = pt_filters_.size();
  vector<float> dists(n);
  for (int i = 0; i < n; ++i) {
    SetKeyPointMeasurementCov(i, std_weight_position_ * mean_scale);
    std::array<float, 2> m{kpts[i].x, kpts[i].y};
    cv::Mat z = as_mat(m);
    auto& f = pt_filters_[i];
    cv::Mat sigma;
    cv::gemm(f.measurementMatrix * f.errorCovPre, f.measurementMatrix, 1, f.measurementNoiseCov, 1,
             sigma, cv::GEMM_2_T);
    cv::Mat r = z - f.measurementMatrix * f.statePre;
    cv::Mat d = r.t() * sigma.inv() * r;
    dists[i] = d.at<float>();
  }
  return dists;
}

void TrackingFilter::SetBboxProcessCov(float sigma_p, float sigma_v) {
  auto& m = bbox_filter_.processNoiseCov;
  cv::setIdentity(m(cv::Rect(0, 0, 4, 4)), sigma_p * sigma_p);
  cv::setIdentity(m(cv::Rect(4, 4, 4, 4)), sigma_v * sigma_v);
}
void TrackingFilter::SetBboxMeasurementCov(float sigma_p) {
  auto& m = bbox_filter_.measurementNoiseCov;
  cv::setIdentity(m, sigma_p * sigma_p);
}
void TrackingFilter::SetBboxErrorCov(float sigma_p, float sigma_v) {
  auto& m = bbox_filter_.errorCovPost;
  cv::setIdentity(m(cv::Rect(0, 0, 4, 4)), sigma_p * sigma_p);
  cv::setIdentity(m(cv::Rect(4, 4, 4, 4)), sigma_v * sigma_v);
}
void TrackingFilter::SetBboxTransitionMat() {
  auto& m = bbox_filter_.transitionMatrix;
  cv::setIdentity(m(cv::Rect(4, 0, 4, 4)));  // with scale velocity
  //  cv::setIdentity(m(cv::Rect(4, 0, 2, 2)));  // w/o scale velocity
}
void TrackingFilter::SetBboxMeasurementMat() {
  auto& m = bbox_filter_.measurementMatrix;
  cv::setIdentity(m(cv::Rect(0, 0, 4, 4)));
}

void TrackingFilter::SetKeyPointProcessCov(int index, float sigma_p, float sigma_v) {
  auto& m = pt_filters_[index].processNoiseCov;
  m.at<float>(0, 0) = sigma_p * sigma_p;
  m.at<float>(1, 1) = sigma_p * sigma_p;
  m.at<float>(2, 2) = sigma_v * sigma_v;
  m.at<float>(3, 3) = sigma_v * sigma_v;
}
void TrackingFilter::SetKeyPointMeasurementCov(int index, float sigma_p) {
  auto& m = pt_filters_[index].measurementNoiseCov;
  m.at<float>(0, 0) = sigma_p * sigma_p;
  m.at<float>(1, 1) = sigma_p * sigma_p;
}
void TrackingFilter::SetKeyPointErrorCov(int index, float sigma_p, float sigma_v) {
  auto& m = pt_filters_[index].errorCovPost;
  m.at<float>(0, 0) = sigma_p * sigma_p;
  m.at<float>(1, 1) = sigma_p * sigma_p;
  m.at<float>(2, 2) = sigma_v * sigma_v;
  m.at<float>(3, 3) = sigma_v * sigma_v;
}
void TrackingFilter::SetKeyPointTransitionMat(int index) {
  auto& m = pt_filters_[index].transitionMatrix;
  cv::setIdentity(m(cv::Rect(2, 0, 2, 2)));
}
void TrackingFilter::SetKeyPointMeasurementMat(int index) {
  auto& m = pt_filters_[index].measurementMatrix;
  cv::setIdentity(m(cv::Rect(0, 0, 2, 2)));
}

void TrackingFilter::ResetKeyPoint(int index, const Point& kpt, float scale) {
  auto& f = pt_filters_[index];
  SetKeyPointErrorCov(index, 2 * std_weight_position_ * scale, 10 * std_weight_velocity_ * scale);
  f.statePost.at<float>(0) = kpt.x;
  f.statePost.at<float>(1) = kpt.y;
}

}  // namespace mmdeploy::mmpose::_pose_tracker
