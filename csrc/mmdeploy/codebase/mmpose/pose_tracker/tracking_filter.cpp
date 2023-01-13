// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_tracker/tracking_filter.h"

namespace mmdeploy::mmpose::_pose_tracker {

TrackingFilter::TrackingFilter(const Bbox& bbox, const vector<Point>& kpts,
                               const TrackingFilter::Params& center_params,
                               const TrackingFilter::Params& scale_params,
                               const TrackingFilter::Params& pts_params) {
  const auto n = kpts.size();
  pt_filters_.resize(n);
  for (int i = 0; i < n; ++i) {
    auto& f = pt_filters_[i];
    auto& p = pts_params;
    f.init(4, 2);
    f.transitionMatrix = pts_trans();
    f.measurementMatrix.at<float>(0, 0) = 1;
    f.measurementMatrix.at<float>(1, 1) = 1;
    f.measurementNoiseCov *= p.measure_sigma * p.measure_sigma;
    f.processNoiseCov = pts_process_cov(p.process_sigma);
    f.statePost.at<float>(0) = kpts[i].x;
    f.statePost.at<float>(1) = kpts[i].y;
  }
  {
    // [x, y, u, v, w, h]
    auto& f = bbox_filter_;
    auto& c = center_params;
    auto& s = scale_params;
    f.init(6, 4);
    f.transitionMatrix = bbox_trans();
    f.measurementMatrix.at<float>(0, 0) = 1;
    f.measurementMatrix.at<float>(1, 1) = 1;
    f.measurementMatrix.at<float>(2, 4) = 1;
    f.measurementMatrix.at<float>(3, 5) = 1;
    f.measurementNoiseCov(cv::Rect(0, 0, 2, 2)) *= c.measure_sigma * c.measure_sigma;
    f.measurementNoiseCov(cv::Rect(2, 2, 2, 2)) *= s.measure_sigma * s.measure_sigma;
    f.processNoiseCov = bbox_process_cov(c.process_sigma, s.process_sigma);
    f.statePost.at<float>(0) = get_center(bbox).x;
    f.statePost.at<float>(1) = get_center(bbox).y;
    f.statePost.at<float>(4) = get_scale(bbox)[0];
    f.statePost.at<float>(5) = get_scale(bbox)[1];
  }
}

std::pair<Bbox, Points> TrackingFilter::Predict() {
  const auto n = pt_filters_.size();
  Points pts(n);
  for (int i = 0; i < n; ++i) {
    auto mat = pt_filters_[i].predict();
    pts[i].x = mat.at<float>(0);
    pts[i].y = mat.at<float>(1);
  }
  Bbox bbox;
  {
    auto mat = bbox_filter_.predict();
    auto x = mat.ptr<float>();
    bbox = get_bbox({x[0], x[1]}, {x[4], x[5]});
  }
  return {bbox, pts};
}

std::pair<Bbox, Points> TrackingFilter::Correct(const Bbox& bbox, const Points& kpts) {
  const auto n = pt_filters_.size();
  Points corr_kpts(n);
  for (int i = 0; i < n; ++i) {
    std::array<float, 2> m{kpts[i].x, kpts[i].y};
    auto mat = pt_filters_[i].correct(cv::Mat(m, false));
    corr_kpts[i].x = mat.at<float>(0);
    corr_kpts[i].y = mat.at<float>(1);
  }
  Bbox corr_bbox;
  {
    auto c = get_center(bbox);
    auto s = get_scale(bbox);
    std::array<float, 4> m{c.x, c.y, s[0], s[1]};
    auto mat = bbox_filter_.correct(cv::Mat(m, false));
    auto x = mat.ptr<float>();
    corr_bbox = get_bbox({x[0], x[1]}, {x[4], x[5]});
  }
  return {corr_bbox, corr_kpts};
}

float TrackingFilter::Distance(const Bbox& bbox) {
  auto c = get_center(bbox);
  auto s = get_scale(bbox);
  std::array<float, 4> m{c.x, c.y, s[0], s[1]};
  cv::Mat z(m, false);
  auto& f = bbox_filter_;
  cv::Mat sigma;
  cv::gemm(f.measurementMatrix * f.errorCovPre, f.measurementMatrix, 1, f.measurementNoiseCov, 1,
           sigma, cv::GEMM_2_T);
  auto r = z - f.measurementMatrix * f.statePre;
  cv::Mat d = r.t() * sigma.inv() * r + std::log(static_cast<float>(cv::determinant(sigma)));
  //  MMDEPLOY_ERROR("{}", d.at<float>(0));
  return 0;
}

const cv::Mat& TrackingFilter::pts_trans() {
  static const cv::Mat trans = [] {
    cv::Mat m = cv::Mat::eye(4, 4, CV_32F);
    m.at<float>(0, 2) = 1;
    m.at<float>(1, 3) = 1;
    return m;
  }();
  return trans;
}

cv::Mat TrackingFilter::pts_process_cov(float sigma) {
  static const cv::Mat proc_cov = [] {
    cv::Mat m = cv::Mat::eye(4, 4, CV_32F);
    cv::setIdentity(m(cv::Rect(0, 0, 2, 2)), .25f);
    cv::setIdentity(m(cv::Rect(0, 2, 2, 2)), .50f);
    cv::setIdentity(m(cv::Rect(2, 0, 2, 2)), .50f);
    return m;
  }();
  return proc_cov * sigma;
}

const cv::Mat& TrackingFilter::bbox_trans() {
  static const cv::Mat trans = [] {
    cv::Mat m = cv::Mat::eye(6, 6, CV_32F);
    pts_trans().copyTo(m(cv::Rect(0, 0, 4, 4)));
    return m;
  }();
  return trans;
}

cv::Mat TrackingFilter::bbox_process_cov(float sigma_c, float sigma_s) {
  static const cv::Mat proc_cov = [] {
    cv::Mat m = cv::Mat::eye(6, 6, CV_32F);
    pts_process_cov(1).copyTo(m(cv::Rect(0, 0, 4, 4)));
    return m;
  }();
  auto m = proc_cov.clone();
  m(cv::Rect(0, 0, 4, 4)) *= sigma_c * sigma_c;
  cv::setIdentity(m(cv::Rect(4, 4, 2, 2)), sigma_s * sigma_s);
  return m;
}

}  // namespace mmdeploy::mmpose::_pose_tracker