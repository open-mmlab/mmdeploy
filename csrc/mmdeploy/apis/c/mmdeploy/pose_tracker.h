// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_POSE_TRACKER_H
#define MMDEPLOY_POSE_TRACKER_H

#include "mmdeploy/common.h"
#include "mmdeploy/detector.h"
#include "mmdeploy/model.h"
#include "mmdeploy/pose_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_pose_tracker* mmdeploy_pose_tracker_t;
typedef struct mmdeploy_pose_tracker_state* mmdeploy_pose_tracker_state_t;

typedef struct mmdeploy_pose_tracker_param_t {
  int32_t det_interval;
  int32_t det_label;
  float det_thr;
  float det_min_bbox_size;
  float det_nms_thr;

  int32_t pose_max_num_bboxes;
  int32_t pose_min_keypoints;
  float pose_min_bbox_size;
  float* keypoint_sigmas;
  int32_t keypoint_sigmas_size;

  float pose_nms_thr;
  float pose_kpt_thr;
  float track_visible_thr;
  float track_missing_thr;
  float track_bbox_scale;
  int32_t track_max_missing;
  int32_t track_history_size;

  float kf_bbox_center[2];
  float kf_bbox_scale[2];
  float kf_key_points[2];

  float smooth_bbox_center[3];
  float smooth_bbox_scale[3];
  float smooth_key_points[3];
} mmdeploy_pose_tracker_param_t;

typedef struct mmdeploy_pose_tracker_result_t {
  mmdeploy_point_t** keypoints;
  float** scores;
  mmdeploy_rect_t* bboxes;
  uint32_t* track_ids;
  int32_t keypoint_count;
  int32_t target_count;
} mmdeploy_pose_tracker_result_t;

MMDEPLOY_API int mmdeploy_pose_tracker_default_params(mmdeploy_pose_tracker_param_t* params);

MMDEPLOY_API int mmdeploy_pose_tracker_create(mmdeploy_model_t det_model,
                                              mmdeploy_model_t pose_model,
                                              mmdeploy_context_t context,
                                              mmdeploy_pose_tracker_t* pipeline);

MMDEPLOY_API void mmdeploy_pose_tracker_destroy(mmdeploy_pose_tracker_t tracker);

MMDEPLOY_API int mmdeploy_pose_tracker_create_state(mmdeploy_pose_tracker_t tracker,
                                                    const mmdeploy_pose_tracker_param_t* params,
                                                    mmdeploy_pose_tracker_state_t* tracker_state);

MMDEPLOY_API void mmdeploy_pose_tracker_destroy_state(mmdeploy_pose_tracker_state_t tracker_state);

MMDEPLOY_API int mmdeploy_pose_tracker_apply(mmdeploy_pose_tracker_t pipeline,
                                             mmdeploy_pose_tracker_state_t* states,
                                             const mmdeploy_mat_t* frames,
                                             const int32_t* use_detect, int32_t batch_size,
                                             mmdeploy_pose_tracker_result_t** results);

MMDEPLOY_API void mmdeploy_pose_tracker_release_result(mmdeploy_pose_tracker_result_t* results,
                                                       int32_t result_count);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_POSE_TRACKER_H
