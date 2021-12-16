// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TEXT_DETECTOR_H
#define MMDEPLOY_TEXT_DETECTOR_H

#include "common.h"

typedef struct mm_text_detect_t {
  mm_pointf_t bbox[4];  ///< a text bounding box of which the vertex are in clock-wise
  float score;
} mm_text_detect_t;

MM_SDK_API int mmdeploy_text_detector_create(mm_model_t model, const char* device_name,
                                             int device_id, mm_handle_t* handle);

/**
 * @brief Create text-detector's handle
 * @param config text-detector's config which is supposed to be json. Refer to
 * `@PROJECT_ROOT_DIR/config/text-detector/dbnet_config.json`
 * @param device_name name of device, such as "cpu", "cuda" and etc.
 * @param device_id id of device.
 * @param handle instance of a text-detector
 * @return status of creating text-detector's handle
 */
MM_SDK_API int mmdeploy_text_detector_create_by_path(const char* model_path,
                                                     const char* device_name, int device_id,
                                                     mm_handle_t* handle);

/**
 * @brief Apply text-detector to batch images and get their inference results
 * @param handle text-detector's handle made by `mmdeploy_text_detector_create_by_path`
 * @param mats batch images
 * @param mat_count number of images in a batch
 * @param results a consecutive buffer to save text detection results of each
 * image. Its length equals to `mat_count` if this function returns success. The buffer should be
 * destroyed by calling `mmdeploy_text_detector_release_result`.
 * @param result_count a consecutive buffer to save the number of detection
 * results of each image. Its length equals to `mat_count` if this function returns success. It
 * should be destroyed by calling `mmdeploy_detector_release_result`
 * @return status of inference
 */
MM_SDK_API int mmdeploy_text_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                            mm_text_detect_t** results, int** result_count);

/** @brief release the inference result buffer
 * @param results a consecutive buffer to save images' text detection results,
 * which is created by `mmdeploy_text_detector_apply`
 * @param result_count a consecutive buffer to save the size of text detection
 * results of each image in a batch. `result_count[i]` represents the text bbox number in the `i`-th
 * input image, i.e. `mats[i]` in `mmdeploy_text_detector_apply`
 * @param count the length of buffer `results`, which is the same as `mat_count` in
 * `mmdeploy_text_detector_apply`
 */
MM_SDK_API void mmdeploy_text_detector_release_result(mm_text_detect_t* results,
                                                      const int* result_count, int count);

/**
 * @brief destroy text-detector's handle
 * @param handle text-detector's handle made by `mmdeploy_text_detector_create_by_path`
 */
MM_SDK_API void mmdeploy_text_detector_destroy(mm_handle_t handle);

#endif  // MMDEPLOY_TEXT_DETECTOR_H
