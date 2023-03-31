//
// Created by zhangli on 1/9/23.
//

#ifndef MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_TRACK_H
#define MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_TRACK_H

#include "pose_tracker/common.h"
#include "pose_tracker/smoothing_filter.h"
#include "pose_tracker/tracking_filter.h"
#include "pose_tracker/utils.h"

namespace mmdeploy::mmpose::_pose_tracker {

class Track {
 public:
  Track(const mmdeploy_pose_tracker_param_t* params, const Bbox& bbox, const Points& kpts,
        const Scores& ss, int64_t id);
  ~Track();

  void UpdateVisible(const Bbox& bbox, const Points& kpts, const Scores& scores,
                     const vector<bool>& tracked);
  void UpdateRecovered(const Bbox& bbox, const Points& kpts, const Scores& scores);
  void UpdateMissing();
  void Predict();

  float BboxDistance(const Bbox& bbox) { return filter_.BboxDistance(bbox); }

  vector<float> KeyPointDistance(const Points& kpts) { return filter_.KeyPointDistance(kpts); }

  uint32_t track_id() const noexcept { return track_id_; }
  uint32_t missing() const noexcept { return missing_; }

  const Bbox& predicted_bbox() const noexcept { return bbox_predict_; }
  const Bbox& smoothed_bbox() const noexcept { return bbox_smooth_; }

  const Points& predicted_kpts() const noexcept { return kpts_predict_; }
  const Points& smoothed_kpts() const noexcept { return kpts_smooth_; }

  const Scores& scores() const noexcept { return scores_.back(); }

 private:
  void Add(const Bbox& bbox, const Points& kpts, const Scores& ss);

  TrackingFilter CreateFilter(const Bbox& bbox, const Points& pts);
  SmoothingFilter CreateSmoother(const Bbox& bbox, const Points& pts);

 private:
  const mmdeploy_pose_tracker_param_t* params_;
  vector<Bbox> bboxes_;
  vector<Points> keypoints_;
  vector<Scores> scores_;
  uint32_t track_id_{};
  Bbox bbox_predict_{};
  Bbox bbox_smooth_{};
  Points kpts_predict_;
  Points kpts_smooth_;
  uint32_t missing_{0};
  TrackingFilter filter_;
  SmoothingFilter smoother_;
};

}  // namespace mmdeploy::mmpose::_pose_tracker

#endif  // MMDEPLOY_TRACK_H
