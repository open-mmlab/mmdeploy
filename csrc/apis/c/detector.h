// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_DETECTOR_H
#define MMDEPLOY_DETECTOR_H

#include "common.h"

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

MM_SDK_API int mmdeploy_detector_create(mm_model_t model, const char* device_name, int device_id,
                                        mm_handle_t* handle);

/**
 * @brief Create detector's handle
 * @param model_path detector's config which is supposed to be json. Refer to
 * `@PROJECT_ROOT_DIR/config/detector/retinanet_config.json`
 * @param device_name name of device, such as "cpu", "cuda" and etc.
 * @param device_id id of device.
 * @param handle instance of a detector
 * @return status of creating detector's handle
 */
MM_SDK_API int mmdeploy_detector_create_by_path(const char* model_path, const char* device_name,
                                                int device_id, mm_handle_t* handle);

/**
 * @brief Apply detector to batch images and get their inference results
 * @param handle detector's handle made by `mmdeploy_detector_create_by_path`
 * @param mats batch images
 * @param mat_count number of images in a batch
 * @param results a consecutive buffer to save detection results of each image.
 * User has to access `result_count` to get the beginning and ending position of
 * an image's detection results. It is created inside and should be destroyed by
 * calling `mmdeploy_detector_release_result`
 * @param result_count a consecutive buffer to save the number of detection
 * results of each image. Its length is equal to `mat_count`. It is created
 * inside and should be destroyed by calling `mmdeploy_detector_release_result`
 * @return status of inference
 */
MM_SDK_API int mmdeploy_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                       mm_detect_t** results, int** result_count);

/** @brief release the inference result buffer
 * @param results a consecutive buffer to save images' detection results
 * @param result_count a consecutive buffer to save the size of detection
 * results of each image in a batch
 * @param count length of `result_count`
 */
MM_SDK_API void mmdeploy_detector_release_result(mm_detect_t* results, const int* result_count,
                                                 int count);

/**
 * @brief destroy detector's handle
 * @param handle detector's handle made by `mmdeploy_detector_create_by_path`
 */
MM_SDK_API void mmdeploy_detector_destroy(mm_handle_t handle);

#endif  // MMDEPLOY_DETECTOR_H
