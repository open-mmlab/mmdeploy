// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_COMMON_H
#define MMDEPLOY_CODEBASE_MMPOSE_POSE_TRACKER_COMMON_H

#include <array>
#include <vector>

#include "mmdeploy/core/mpl/type_traits.h"
#include "mmdeploy/pose_tracker.h"

namespace mmdeploy::mmpose::_pose_tracker {

struct TrackerResult {
  std::vector<std::vector<mmdeploy_point_t>> keypoints;
  std::vector<std::vector<float>> scores;
  std::vector<mmdeploy_rect_t> bboxes;
  std::vector<uint32_t> track_ids;
  // debug info
  std::vector<std::array<float, 4>> pose_input_bboxes;
  std::vector<std::array<float, 4>> pose_output_bboxes;
};

inline void SetDefaultParams(mmdeploy_pose_tracker_param_t& p) {
  p.det_interval = 1;
  p.det_label = 0;
  p.det_min_bbox_size = -1;
  p.det_thr = .5f;
  p.det_nms_thr = .7f;
  p.pose_max_num_bboxes = -1;
  p.pose_min_keypoints = -1;
  p.pose_min_bbox_size = 0;
  p.pose_kpt_thr = .5f;
  p.pose_nms_thr = .5f;
  p.keypoint_sigmas = nullptr;
  p.keypoint_sigmas_size = 0;
  p.track_iou_thr = .4f;
  p.pose_bbox_scale = 1.25f;
  p.track_max_missing = 10;
  p.track_history_size = 1;

  p.std_weight_position = 1 / 20.f;
  p.std_weight_velocity = 1 / 160.f;

  (std::array<float, 3>&)p.smooth_params = {0.007, 1., 1.};
}

}  // namespace mmdeploy::mmpose::_pose_tracker

namespace mmdeploy {

MMDEPLOY_REGISTER_TYPE_ID(mmdeploy_pose_tracker_param_t*, 0x32bc6919d5287035);
MMDEPLOY_REGISTER_TYPE_ID(mmpose::_pose_tracker::TrackerResult, 0xacb6ddb7dc1ffbca);

}  // namespace mmdeploy

#endif  // MMDEPLOY_COMMON_H
