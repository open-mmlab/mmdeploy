// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file text_detector.h
 * @brief Interface to MMOCR text detection task
 */

#ifndef MMDEPLOY_TEXT_DETECTOR_H
#define MMDEPLOY_TEXT_DETECTOR_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mm_text_detect_t {
  mm_pointf_t bbox[4];  ///< a text bounding box of which the vertex are in clock-wise
  float score;
} mm_text_detect_t;

/**
 * @brief Create text-detector's handle
 * @param[in] model an instance of mmocr text detection model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle instance of a text-detector, which must be destroyed
 * by \ref mmdeploy_text_detector_destroy
 * @return status of creating text-detector's handle
 */
MMDEPLOY_API int mmdeploy_text_detector_create(mm_model_t model, const char* device_name,
                                               int device_id, mm_handle_t* handle);

/**
 * @brief Create text-detector's handle
 * @param[in] model_path path to text detection model
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device
 * @param[out] handle instance of a text-detector, which must be destroyed
 * by \ref mmdeploy_text_detector_destroy
 * @return status of creating text-detector's handle
 */
MMDEPLOY_API int mmdeploy_text_detector_create_by_path(const char* model_path,
                                                       const char* device_name, int device_id,
                                                       mm_handle_t* handle);

/**
 * @brief Apply text-detector to batch images and get their inference results
 * @param[in] handle text-detector's handle created by \ref mmdeploy_text_detector_create_by_path
 * @param[in] mats a batch of images
 * @param[in] mat_count number of images in the batch
 * @param[out] results a linear buffer to save text detection results of each
 * image. It must be released by calling \ref mmdeploy_text_detector_release_result
 * @param[out] result_count a linear buffer of length \p mat_count to save the number of detection
 * results of each image. It must be released by \ref mmdeploy_detector_release_result
 * @return status of inference
 */
MMDEPLOY_API int mmdeploy_text_detector_apply(mm_handle_t handle, const mm_mat_t* mats,
                                              int mat_count, mm_text_detect_t** results,
                                              int** result_count);

/** @brief Release the inference result buffer returned by \ref mmdeploy_text_detector_apply
 * @param[in] results text detection result buffer
 * @param[in] result_count  \p results size buffer
 * @param[in] count the length of buffer \p result_count
 */
MMDEPLOY_API void mmdeploy_text_detector_release_result(mm_text_detect_t* results,
                                                        const int* result_count, int count);

/**
 * @brief Destroy text-detector's handle
 * @param[in] handle text-detector's handle created by \ref mmdeploy_text_detector_create_by_path or
 * \ref mmdeploy_text_detector_create
 */
MMDEPLOY_API void mmdeploy_text_detector_destroy(mm_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_TEXT_DETECTOR_H
