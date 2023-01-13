// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_TRACKING_FILTER_H
#define MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_TRACKING_FILTER_H

#include "opencv2/video/video.hpp"
#include "pose_tracker/utils.h"


namespace mmdeploy::mmpose::_pose_tracker {

// use Kalman filter to estimate and predict true states
class TrackingFilter {
 public:
  struct Params {
    float measure_sigma;
    float process_sigma;
  };
  TrackingFilter(const Bbox& bbox, const vector<Point>& kpts, const Params& center_params,
                 const Params& scale_params, const Params& pts_params);

  std::pair<Bbox, Points> Predict();

  float Distance(const Bbox& bbox);

  std::pair<Bbox, Points> Correct(const Bbox& bbox, const Points& kpts);

 private:
  static const cv::Mat& pts_trans();

  static cv::Mat pts_process_cov(float sigma);

  static const cv::Mat& bbox_trans();

  static cv::Mat bbox_process_cov(float sigma_c, float sigma_s);



 private:
  std::vector<cv::KalmanFilter> pt_filters_;
  cv::KalmanFilter bbox_filter_;
};

}  // namespace mmdeploy::mmpose::_pose_tracker

#endif  // MMDEPLOY_TRACKING_FILTER_H
