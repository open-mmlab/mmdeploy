// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_tracker/track.h"

namespace mmdeploy::mmpose::_pose_tracker {

Track::Track(const mmdeploy_pose_tracker_param_t* params, const Bbox& bbox, const Points& kpts,
             const Scores& ss, int64_t id)
    : params_(params),
      filter_(CreateFilter(bbox, kpts)),
      smoother_(CreateSmoother(bbox, kpts)),
      track_id_(id) {
  POSE_TRACKER_DEBUG("new track {}", track_id_);
  Add(bbox, kpts, ss);
}

Track::~Track() { POSE_TRACKER_DEBUG("track lost {}", track_id_); }

void Track::UpdateVisible(const Bbox& bbox, const Points& kpts, const Scores& scores,
                          const vector<bool>& tracked) {
  auto [bbox_corr, kpts_corr] = filter_.Correct(bbox, kpts, tracked);
  Add(bbox_corr, kpts_corr, scores);
}

void Track::UpdateRecovered(const Bbox& bbox, const Points& kpts, const Scores& scores) {
  POSE_TRACKER_DEBUG("track recovered {}", track_id_);
  filter_ = CreateFilter(bbox, kpts);
  smoother_.Reset(bbox, kpts);
  Add(bbox, kpts, scores);
  missing_ = 0;
}

void Track::UpdateMissing() {
  missing_++;
  if (missing_ <= params_->track_max_missing) {
    // use predicted state to update the missing tracks
    Add(bbox_predict_, kpts_predict_, vector<float>(kpts_predict_.size()));
  }
}

void Track::Predict() {
  // TODO: velocity decay for missing tracks
  std::tie(bbox_predict_, kpts_predict_) = filter_.Predict();
}

void Track::Add(const Bbox& bbox, const Points& kpts, const Scores& ss) {
  bboxes_.push_back(bbox);
  keypoints_.push_back(kpts);
  scores_.push_back(ss);
  if (bboxes_.size() > params_->track_history_size) {
    std::rotate(bboxes_.begin(), bboxes_.begin() + 1, bboxes_.end());
    std::rotate(keypoints_.begin(), keypoints_.begin() + 1, keypoints_.end());
    std::rotate(scores_.begin(), scores_.begin() + 1, scores_.end());
    bboxes_.pop_back();
    keypoints_.pop_back();
    scores_.pop_back();
  }
  std::tie(bbox_smooth_, kpts_smooth_) = smoother_.Step(bbox, kpts);
}

TrackingFilter Track::CreateFilter(const Bbox& bbox, const Points& pts) {
  return {bbox, pts, params_->std_weight_position, params_->std_weight_velocity};
}

SmoothingFilter Track::CreateSmoother(const Bbox& bbox, const Points& pts) {
  return SmoothingFilter(
      bbox, pts, {params_->smooth_params[0], params_->smooth_params[1], params_->smooth_params[2]});
}

}  // namespace mmdeploy::mmpose::_pose_tracker
