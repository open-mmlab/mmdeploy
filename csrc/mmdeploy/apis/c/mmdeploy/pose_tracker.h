// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file pose_tracker.h
 * @brief Pose tracker C API
 */

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
  // detection interval, default = 1
  int32_t det_interval;
  // detection label use for pose estimation, default = 0
  int32_t det_label;
  // detection score threshold, default = 0.5
  float det_thr;
  // detection minimum bbox size (compute as sqrt(area)), default = -1
  float det_min_bbox_size;
  // nms iou threshold for merging detected bboxes and bboxes from tracked targets, default = 0.7
  float det_nms_thr;

  // max number of bboxes used for pose estimation per frame, default = -1
  int32_t pose_max_num_bboxes;
  // threshold for visible key-points, default = 0.5
  float pose_kpt_thr;
  // min number of key-points for valid poses (-1 indicates ceil(n_kpts/2)), default = -1
  int32_t pose_min_keypoints;
  // scale for expanding key-points to bbox, default = 1.25
  float pose_bbox_scale;
  // min pose bbox size, tracks with bbox size smaller than the threshold will be dropped,
  // default = -1
  float pose_min_bbox_size;
  // nms oks/iou threshold for suppressing overlapped poses, useful when multiple pose estimations
  // collapse to the same target, default = 0.5
  float pose_nms_thr;
  // keypoint sigmas for computing OKS, will use IOU if not set, default = nullptr
  float* keypoint_sigmas;
  // size of keypoint sigma array, must be consistent with the number of key-points, default = 0
  int32_t keypoint_sigmas_size;

  // iou threshold for associating missing tracks, default = 0.4
  float track_iou_thr;
  // max number of missing frames before a missing tracks is removed, default = 10
  int32_t track_max_missing;
  // track history size, default = 1
  int32_t track_history_size;

  // weight of position for setting covariance matrices of kalman filters, default = 0.05
  float std_weight_position;
  // weight of velocity for setting covariance matrices of kalman filters, default = 0.00625
  float std_weight_velocity;

  // params for the one-euro filter for smoothing the outputs - (beta, fc_min, fc_derivative)
  // default = (0.007, 1, 1)
  float smooth_params[3];
} mmdeploy_pose_tracker_param_t;

typedef struct mmdeploy_pose_tracker_target_t {
  mmdeploy_point_t* keypoints;  // key-points of the target
  int32_t keypoint_count;       // size of `keypoints` array
  float* scores;                // scores of each key-point
  mmdeploy_rect_t bbox;         // estimated bbox from key-points
  uint32_t target_id;           // target id from internal tracker
} mmdeploy_pose_tracker_target_t;

/**
 * @brief Fill params with default parameters
 * @param[in,out] params
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pose_tracker_default_params(mmdeploy_pose_tracker_param_t* params);

/**
 * @brief Create pose tracker pipeline
 * @param[in] det_model detection model object, created by \ref mmdeploy_model_create
 * @param[in] pose_model pose model object
 * @param[in] context context object describing execution environment (device, profiler, etc...),
 * created by \ref mmdeploy_context_create
 * @param[out] pipeline handle of the created pipeline
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pose_tracker_create(mmdeploy_model_t det_model,
                                              mmdeploy_model_t pose_model,
                                              mmdeploy_context_t context,
                                              mmdeploy_pose_tracker_t* pipeline);

/**
 * @brief Destroy pose tracker pipeline
 * @param[in] pipeline
 */
MMDEPLOY_API void mmdeploy_pose_tracker_destroy(mmdeploy_pose_tracker_t pipeline);

/**
 * @brief Create a tracker state handle corresponds to a video stream
 * @param[in] pipeline handle of a pose tracker pipeline
 * @param[in] params params for creating the tracker state
 * @param[out] state handle of the created tracker state
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pose_tracker_create_state(mmdeploy_pose_tracker_t pipeline,
                                                    const mmdeploy_pose_tracker_param_t* params,
                                                    mmdeploy_pose_tracker_state_t* state);

/**
 * @brief Destroy tracker state
 * @param[in] state handle of the tracker state
 */
MMDEPLOY_API void mmdeploy_pose_tracker_destroy_state(mmdeploy_pose_tracker_state_t state);

/**
 * @brief Apply pose tracker pipeline, notice that this function supports batch operation by feeding
 * arrays of size \p count to \p states, \p frames and \p use_detect
 * @param[in] pipeline handle of a pose tracker pipeline
 * @param[in] states tracker states handles, array of size \p count
 * @param[in] frames input frames of size \p count
 * @param[in] use_detect control the use of detector, array of size \p count
 *   -1: use params.det_interval, 0: don't use detector, 1: force use detector
 * @param[in] count batch size
 * @param[out] results a linear buffer contains the tracked targets of input frames. Should be
 * released by \ref mmdeploy_pose_tracker_release_result
 * @param[out] result_count a linear buffer of size \p count contains the number of tracked
 * targets of the frames. Should be released by \ref mmdeploy_pose_tracker_release_result
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pose_tracker_apply(mmdeploy_pose_tracker_t pipeline,
                                             mmdeploy_pose_tracker_state_t* states,
                                             const mmdeploy_mat_t* frames,
                                             const int32_t* use_detect, int32_t count,
                                             mmdeploy_pose_tracker_target_t** results,
                                             int32_t** result_count);

/**
 * @brief Release result objects
 * @param[in] results
 * @param[in] result_count
 * @param[in] count
 */
MMDEPLOY_API void mmdeploy_pose_tracker_release_result(mmdeploy_pose_tracker_target_t* results,
                                                       const int32_t* result_count, int count);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_POSE_TRACKER_H
