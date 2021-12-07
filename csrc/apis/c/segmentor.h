// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SEGMENTOR_H
#define MMDEPLOY_SEGMENTOR_H

#include "common.h"

typedef struct mm_segment_t {
  int height;   ///< height of `mask` that equals to the input image's height
  int width;    ///< width of `mask` that equals to the input image's width
  int classes;  ///< the number of labels in `mask`
  int* mask;    ///< segmentation mask of the input image, in which `mask[i * width + j]` indicates
                ///< the `label_id` of pixel at `(i, j)`
} mm_segment_t;

MM_SDK_API int mmdeploy_segmentor_create(mm_model_t model, const char* device_name, int device_id,
                                         mm_handle_t* handle);

/**
 * @brief Create segmentor's handle
 * @param model_path segmentor's config which is supposed to be json. Refer to
 * `@PROJECT_ROOT_DIR/config/segmentor/fcn_config.json`
 * @param device_name name of device, such as "cpu", "cuda" and etc.
 * @param device_id id of device.
 * @param handle instance of a segmentor
 * @return status of creating segmentor's handle
 */
MM_SDK_API int mmdeploy_segmentor_create_by_path(const char* model_path, const char* device_name,
                                                 int device_id, mm_handle_t* handle);

/**
 * @brief Apply segmentor to batch images and get their inference results
 * @param handle segmentor's handle made by `mmdeploy_segmentor_create_by_path`
 * @param mats batch images
 * @param mat_count number of images in a batch
 * @param results a consecutive buffer to save segmentation results of each
 * image. The length of `results` equals to `mat_count` if this function returns success. It should
 * be destroyed by calling `mmdeploy_segmentor_release_result`.
 * @return status of inference
 */
MM_SDK_API int mmdeploy_segmentor_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                        mm_segment_t** results);

/** @brief release the inference result buffer
 * @param results a consecutive buffer to save images' segmentation results,
 * which is created by `mmdeploy_segmentor_apply`
 * @param count the length of buffer `results` that equals to `mat_count` in
 * `mmdeploy_segmentor_apply`
 */
MM_SDK_API void mmdeploy_segmentor_release_result(mm_segment_t* results, int count);

/**
 * @brief destroy segmentor's handle
 * @param handle segmentor's handle made by `mmdeploy_segmentor_create_by_path`
 */
MM_SDK_API void mmdeploy_segmentor_destroy(mm_handle_t handle);

#endif  // MMDEPLOY_SEGMENTOR_H
