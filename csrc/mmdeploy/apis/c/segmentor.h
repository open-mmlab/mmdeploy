// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file segmentor.h
 * @brief Interface to MMSegmentation task
 */

#ifndef MMDEPLOY_SEGMENTOR_H
#define MMDEPLOY_SEGMENTOR_H

#include "common.h"
#include "executor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mm_segment_t {
  int height;   ///< height of \p mask that equals to the input image's height
  int width;    ///< width of \p mask that equals to the input image's width
  int classes;  ///< the number of labels in \p mask
  int* mask;    ///< segmentation mask of the input image, in which mask[i * width + j] indicates
                ///< the label id of pixel at (i, j)
} mm_segment_t;

/**
 * @brief Create segmentor's handle
 * @param[in] model an instance of mmsegmentation sdk model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle instance of a segmentor, which must be destroyed
 * by \ref mmdeploy_segmentor_destroy
 * @return status of creating segmentor's handle
 */
MMDEPLOY_API int mmdeploy_segmentor_create(mm_model_t model, const char* device_name, int device_id,
                                           mm_handle_t* handle);

/**
 * @brief Create segmentor's handle
 * @param[in] model_path path of mmsegmentation sdk model exported by mmdeploy model converter
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle instance of a segmentor, which must be destroyed
 * by \ref mmdeploy_segmentor_destroy
 * @return status of creating segmentor's handle
 */
MMDEPLOY_API int mmdeploy_segmentor_create_by_path(const char* model_path, const char* device_name,
                                                   int device_id, mm_handle_t* handle);

/**
 * @brief Apply segmentor to batch images and get their inference results
 * @param[in] handle segmentor's handle created by \ref mmdeploy_segmentor_create_by_path or \ref
 * mmdeploy_segmentor_create
 * @param[in] mats a batch of images
 * @param[in] mat_count number of images in the batch
 * @param[out] results a linear buffer of length \p mat_count to save segmentation result of each
 * image. It must be released by \ref mmdeploy_segmentor_release_result
 * @return status of inference
 */
MMDEPLOY_API int mmdeploy_segmentor_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                          mm_segment_t** results);

/**
 * @brief Release result buffer returned by \ref mmdeploy_segmentor_apply
 * @param[in] results result buffer
 * @param[in] count length of \p results
 */
MMDEPLOY_API void mmdeploy_segmentor_release_result(mm_segment_t* results, int count);

/**
 * @brief Destroy segmentor's handle
 * @param[in] handle segmentor's handle created by \ref mmdeploy_segmentor_create_by_path
 */
MMDEPLOY_API void mmdeploy_segmentor_destroy(mm_handle_t handle);

/******************************************************************************
 * Experimental asynchronous APIs */

MMDEPLOY_API int mmdeploy_segmentor_create_v2(mm_model_t model, const char* device_name,
                                              int device_id, mmdeploy_exec_info_t exec_info,
                                              mm_handle_t* handle);

MMDEPLOY_API int mmdeploy_segmentor_create_input(const mm_mat_t* mats, int mat_count,
                                                 mmdeploy_value_t* value);

MMDEPLOY_API int mmdeploy_segmentor_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                             mmdeploy_value_t* output);

MMDEPLOY_API int mmdeploy_segmentor_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                                mmdeploy_sender_t* output);

MMDEPLOY_API int mmdeploy_segmentor_get_result(mmdeploy_value_t output, mm_segment_t** results);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_SEGMENTOR_H
