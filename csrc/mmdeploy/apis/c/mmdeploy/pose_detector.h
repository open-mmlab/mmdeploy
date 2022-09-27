// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file pose_detector.h
 * @brief Interface to MMPose task
 */

#ifndef MMDEPLOY_SRC_APIS_C_POSE_DETECTOR_H_
#define MMDEPLOY_SRC_APIS_C_POSE_DETECTOR_H_

#include "common.h"
#include "executor.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_pose_detection_t {
  mmdeploy_point_t* point;  ///< keypoint
  float* score;             ///< keypoint score
  int length;               ///< number of keypoint
} mmdeploy_pose_detection_t;

typedef struct mmdeploy_pose_detector* mmdeploy_pose_detector_t;

/**
 * @brief Create a pose detector instance
 * @param[in] model an instance of mmpose model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] detector handle of the created pose detector, which must be destroyed
 * by \ref mmdeploy_pose_detector_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_pose_detector_create(mmdeploy_model_t model, const char* device_name,
                                               int device_id, mmdeploy_pose_detector_t* detector);

/**
 * @brief Create a pose detector instance
 * @param[in] model_path path to pose detection model
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] detector handle of the created pose detector, which must be destroyed
 * by \ref mmdeploy_pose_detector_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_pose_detector_create_by_path(const char* model_path,
                                                       const char* device_name, int device_id,
                                                       mmdeploy_pose_detector_t* detector);

/**
 * @brief Apply pose detector to a batch of images with full image roi
 * @param[in] detector pose detector's handle created by \ref
 * mmdeploy_pose_detector_create_by_path
 * @param[in] images a batch of images
 * @param[in] count number of images in the batch
 * @param[out] results a linear buffer contains the pose result, must be release
 * by \ref mmdeploy_pose_detector_release_result
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_pose_detector_apply(mmdeploy_pose_detector_t detector,
                                              const mmdeploy_mat_t* mats, int mat_count,
                                              mmdeploy_pose_detection_t** results);

/**
 * @brief Apply pose detector to a batch of images supplied with bboxes(roi)
 * @param[in] detector pose detector's handle created by \ref
 * mmdeploy_pose_detector_create_by_path
 * @param[in] images a batch of images
 * @param[in] image_count number of images in the batch
 * @param[in] bboxes bounding boxes(roi) detected by mmdet
 * @param[in] bbox_count number of bboxes of each \p images, must be same length as \p images
 * @param[out] results a linear buffer contains the pose result, which has the same length as \p
 * bboxes, must be release by \ref mmdeploy_pose_detector_release_result
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_pose_detector_apply_bbox(mmdeploy_pose_detector_t detector,
                                                   const mmdeploy_mat_t* mats, int mat_count,
                                                   const mmdeploy_rect_t* bboxes,
                                                   const int* bbox_count,
                                                   mmdeploy_pose_detection_t** results);

/** @brief Release result buffer returned by \ref mmdeploy_pose_detector_apply or \ref
 * mmdeploy_pose_detector_apply_bbox
 * @param[in] results result buffer by pose detector
 * @param[in] count length of \p result
 */
MMDEPLOY_API void mmdeploy_pose_detector_release_result(mmdeploy_pose_detection_t* results,
                                                        int count);

/**
 * @brief destroy pose_detector
 * @param[in] detector handle of pose_detector created by \ref
 * mmdeploy_pose_detector_create_by_path or \ref mmdeploy_pose_detector_create
 */
MMDEPLOY_API void mmdeploy_pose_detector_destroy(mmdeploy_pose_detector_t detector);

/******************************************************************************
 * Experimental asynchronous APIs */

MMDEPLOY_API int mmdeploy_pose_detector_create_v2(mmdeploy_model_t model,
                                                  mmdeploy_context_t context,
                                                  mmdeploy_pose_detector_t* detector);

MMDEPLOY_API int mmdeploy_pose_detector_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                                     const mmdeploy_rect_t* bboxes,
                                                     const int* bbox_count,
                                                     mmdeploy_value_t* value);

MMDEPLOY_API int mmdeploy_pose_detector_apply_v2(mmdeploy_pose_detector_t detector,
                                                 mmdeploy_value_t input, mmdeploy_value_t* output);

MMDEPLOY_API int mmdeploy_pose_detector_apply_async(mmdeploy_pose_detector_t detector,
                                                    mmdeploy_sender_t input,
                                                    mmdeploy_sender_t* output);

MMDEPLOY_API int mmdeploy_pose_detector_get_result(mmdeploy_value_t output,
                                                   mmdeploy_pose_detection_t** results);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_SRC_APIS_C_POSE_DETECTOR_H_
