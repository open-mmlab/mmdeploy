// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_TRACKING_FILTER_H
#define MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_TRACKING_FILTER_H

#include "opencv2/video/video.hpp"
#include "pose_tracker/utils.h"

namespace mmdeploy::mmpose::_pose_tracker {

// use Kalman filter to estimate and predict true states
class TrackingFilter {
 public:
  TrackingFilter(const Bbox& bbox, const vector<Point>& kpts, float std_weight_position,
                 float std_weight_velocity);

  std::pair<Bbox, Points> Predict();

  vector<float> KeyPointDistance(const Points& kpts);

  float BboxDistance(const Bbox& bbox);

  std::pair<Bbox, Points> Correct(const Bbox& bbox, const Points& kpts,
                                  const vector<bool>& tracked);

 private:
  void SetBboxProcessCov(float sigma_p, float sigma_v);
  void SetBboxMeasurementCov(float sigma_p);
  void SetBboxErrorCov(float sigma_p, float sigma_v);
  void SetBboxTransitionMat();
  void SetBboxMeasurementMat();

  void SetKeyPointProcessCov(int index, float sigma_p, float sigma_v);
  void SetKeyPointMeasurementCov(int index, float sigma_p);
  void SetKeyPointErrorCov(int index, float sigma_p, float sigma_v);
  void SetKeyPointTransitionMat(int index);
  void SetKeyPointMeasurementMat(int index);

  void ResetKeyPoint(int index, const Point& kpt, float scale);

 private:
  std::vector<cv::KalmanFilter> pt_filters_;
  cv::KalmanFilter bbox_filter_;
  float std_weight_position_;
  float std_weight_velocity_;
};

}  // namespace mmdeploy::mmpose::_pose_tracker

#endif  // MMDEPLOY_TRACKING_FILTER_H
