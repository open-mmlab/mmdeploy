// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_HPP
#define MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_HPP

#include "mmdeploy/pose_tracker.h"
#include "pose_tracker/common.h"
#include "pose_tracker/track.h"

namespace mmdeploy::mmpose::_pose_tracker {

class Tracker {
 public:
  explicit Tracker(const mmdeploy_pose_tracker_param_t& _params);

  Tracker(const Tracker&) { assert(0); }
  Tracker(Tracker&& o) noexcept = default;

  struct Detections {
    Bboxes bboxes;
    Scores scores;
    vector<int> labels;
  };

  void SetFrameSize(int height, int width) {
    frame_h_ = static_cast<float>(height);
    frame_w_ = static_cast<float>(width);
  }

  const mmdeploy_pose_tracker_param_t& params() const noexcept { return params_; }

  int64_t frame_id() const noexcept { return frame_id_; }

  const vector<std::unique_ptr<Track>>& tracks() const noexcept { return tracks_; }

  std::tuple<vector<Bbox>, vector<int64_t>> ProcessBboxes(const std::optional<Detections>& dets);

  void TrackStep(vector<Points>& keypoints, vector<Scores>& scores,
                 const vector<int64_t>& prev_ids) noexcept;

 private:
  void GetDetectedObjects(const std::optional<Detections>& dets, vector<Bbox>& _bboxes,
                          vector<int64_t>& track_ids, vector<int>& types) const;

  void GetTrackedObjects(vector<Bbox>& bboxes, vector<int64_t>& track_ids,
                         vector<int>& types) const;

  void SuppressOverlappingBoxes(const vector<Bbox>& bboxes, vector<std::pair<int, float>>& ranks,
                                vector<int>& is_valid_bbox) const;

  void SuppressOverlappingPoses(const vector<Points>& keypoints, const vector<Scores>& scores,
                                const vector<Bbox>& bboxes, const vector<int64_t>& track_ids,
                                vector<int>& is_valid, float oks_thr);

  std::optional<Bbox> KeypointsToBbox(const Points& kpts, const Scores& scores) const;

  float GetPosePoseSimilarity(const Bbox& bbox0, const Points& kpts0, const Bbox& bbox1,
                              const Points& kpts1);

  void GetSimilarityMatrices(const vector<Bbox>& bboxes, const vector<Points>& keypoints,
                             const vector<int64_t>& prev_ids, vector<float>& similarity0,
                             vector<float>& similarity1, vector<vector<bool>>& gating);

  std::tuple<float, float, vector<bool>> GetTrackPoseSimilarity(Track& track, const Bbox& bbox,
                                                                const Points& kpts) const;

  void CreateTrack(const Bbox& bbox, const Points& kpts, const Scores& scores);

  void RemoveMissingTracks();

  void DiagnosticMissingTracks(const vector<int>& is_unused_track,
                               const vector<int>& is_unused_bbox, const vector<float>& similarity0,
                               const vector<float>& similarity1);

  void SummaryTracks();

 private:
  static constexpr const auto kInf = 1000.f;

  float frame_h_ = 0;
  float frame_w_ = 0;

  vector<std::unique_ptr<Track>> tracks_;
  int64_t next_id_{0};

  std::vector<float> key_point_sigmas_;
  mmdeploy_pose_tracker_param_t params_;

  vector<Bbox> pose_input_bboxes_;
  vector<Bbox> pose_output_bboxes_;

  int64_t frame_id_ = 0;

 public:
  const vector<Bbox>& pose_input_bboxes() const noexcept { return pose_input_bboxes_; }
  const vector<Bbox>& pose_output_bboxes() const noexcept { return pose_output_bboxes_; }
};

}  // namespace mmdeploy::mmpose::_pose_tracker

#endif  // MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_HPP
