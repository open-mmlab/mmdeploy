// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CLASSIFIER_H
#define MMDEPLOY_CLASSIFIER_H

#include "common.h"

typedef struct mm_class_t {
  int label_id;
  float score;
} mm_class_t;

MM_SDK_API int mmdeploy_classifier_create(mm_model_t model, const char* device_name, int device_id,
                                          mm_handle_t* handle);

/**
 * @brief Create classifier's handle
 * @param config classifier's config which is supposed to be json
 * @param device_name name of device, such as "cpu", "cuda" and etc.
 * @param device_id id of device.
 * @param handle instance of a classifier
 * @return status of creating classifier's handle
 */
MM_SDK_API int mmdeploy_classifier_create_by_path(const char* model_path, const char* device_name,
                                                  int device_id, mm_handle_t* handle);

/**
 * @brief Use classifier created by `mmdeploy_classifier_create_by_path` to get label information
 * of each image in a batch
 * @param handle classifier's handle made by `mmdeploy_classifier_create_by_path`
 * @param mats batch images
 * @param mat_count number of images in a batch
 * @param results a consecutive buffer to save classification results of each
 * image. User has to access `result_count` to get the range of an
 * image's classification result. `results` is created inside, and should be
 * freed by calling `mmdeploy_classifier_release_result`
 * @param result_count a consecutive buffer to save the number of classification
 * results of each image. Its length is equal to `mat_count`. It is created
 * inside and should be destroyed by calling `mmdeploy_classifier_release_result`
 * @return status of inference
 */
MM_SDK_API int mmdeploy_classifier_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                         mm_class_t** results, int** result_count);

/**
 * @brief release the inference result buffer
 * @param results a consecutive buffer to save images' classification results
 * @param result_count a consecutive buffer to save the size of classification
 * results of each image in a batch
 * @param count length of `result_count`
 */
MM_SDK_API void mmdeploy_classifier_release_result(mm_class_t* results, const int* result_count,
                                                   int count);

/**
 * @brief destroy classifier's handle
 * @param handle classifier's handle made by `mmdeploy_classifier_create_by_path`
 */
MM_SDK_API void mmdeploy_classifier_destroy(mm_handle_t handle);

#endif  // MMDEPLOY_CLASSIFIER_H
