// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file detector.h
 * @brief Interface to MMDetection task
 */

#ifndef MMDEPLOY_DETECTOR_H
#define MMDEPLOY_DETECTOR_H

#include "common.h"
#include "executor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mm_instance_mask_t {
  char* data;
  int height;
  int width;
} mm_instance_mask_t;

typedef struct mm_detect_t {
  int label_id;
  float score;
  mm_rect_t bbox;
  mm_instance_mask_t* mask;
} mm_detect_t;

/**
 * @brief Create detector's handle
 * @param[in] model an instance of mmdetection sdk model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle instance of a detector
 * @return status of creating detector's handle
 */
MMDEPLOY_API int mmdeploy_detector_create(mm_model_t model, const char* device_name, int device_id,
                                          mm_handle_t* handle);

/**
 * @brief Create detector's handle
 * @param[in] model_path path of mmdetection sdk model exported by mmdeploy model converter
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle instance of a detector
 * @return status of creating detector's handle
 */
MMDEPLOY_API int mmdeploy_detector_create_by_path(const char* model_path, const char* device_name,
                                                  int device_id, mm_handle_t* handle);

/**
 * @brief Apply detector to batch images and get their inference results
 * @param[in] handle detector's handle created by \ref mmdeploy_detector_create_by_path
 * @param[in] mats a batch of images
 * @param[in] mat_count number of images in the batch
 * @param[out] results a linear buffer to save detection results of each image. It must be released
 * by \ref mmdeploy_detector_release_result
 * @param[out] result_count a linear buffer with length being \p mat_count to save the number of
 * detection results of each image. And it must be released by \ref
 * mmdeploy_detector_release_result
 * @return status of inference
 */
MMDEPLOY_API int mmdeploy_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                         mm_detect_t** results, int** result_count);

/** @brief Release the inference result buffer created by \ref mmdeploy_detector_apply
 * @param[in] results detection results buffer
 * @param[in] result_count  \p results size buffer
 * @param[in] count length of \p result_count
 */
MMDEPLOY_API void mmdeploy_detector_release_result(mm_detect_t* results, const int* result_count,
                                                   int count);

/**
 * @brief Destroy detector's handle
 * @param[in] handle detector's handle created by \ref mmdeploy_detector_create_by_path
 */
MMDEPLOY_API void mmdeploy_detector_destroy(mm_handle_t handle);

/******************************************************************************
 * Experimental asynchronous APIs */

/**
 * @brief Same as \ref mmdeploy_detector_create, but allows to control execution context of tasks
 * via exec_info
 */
MMDEPLOY_API int mmdeploy_detector_create_v2(mm_model_t model, const char* device_name,
                                             int device_id, mmdeploy_exec_info_t exec_info,
                                             mm_handle_t* handle);

/**
 * @brief Pack detector inputs into mmdeploy_value_t
 * @param[in] mats a batch of images
 * @param[in] mat_count number of images in the batch
 * @return the created value
 */
MMDEPLOY_API int mmdeploy_detector_create_input(const mm_mat_t* mats, int mat_count,
                                                mmdeploy_value_t* input);

/**
 * @brief Same as \ref mmdeploy_detector_apply, but input and output are packed in \ref
 * mmdeploy_value_t.
 */
MMDEPLOY_API int mmdeploy_detector_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                            mmdeploy_value_t* output);

/**
 * @brief Apply detector asynchronously
 * @param[in] handle handle to the detector
 * @param[in] input input sender
 * @return output sender
 */
MMDEPLOY_API int mmdeploy_detector_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                               mmdeploy_sender_t* output);

/**
 * @brief Unpack detector output from a mmdeploy_value_t
 * @param[in] output output obtained by applying a detector
 * @param[out] results a linear buffer to save detection results of each image. It must be released
 * by \ref mmdeploy_detector_release_result
 * @param[out] result_count a linear buffer with length number of input images to save the number of
 * detection results of each image. Must be released by \ref
 * mmdeploy_detector_release_result
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_detector_get_result(mmdeploy_value_t output, mm_detect_t** results,
                                              int** result_count);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_DETECTOR_H
