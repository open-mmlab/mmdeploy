// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file restorer.h
 * @brief Interface to MMEditing image restoration task
 */

#ifndef MMDEPLOY_SRC_APIS_C_RESTORER_H_
#define MMDEPLOY_SRC_APIS_C_RESTORER_H_

#include "common.h"
#include "executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a restorer instance
 * @param[in] model an instance of image restoration model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle handle of the created restorer, which must be destroyed
 * by \ref mmdeploy_restorer_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_restorer_create(mm_model_t model, const char* device_name, int device_id,
                                          mm_handle_t* handle);

/**
 * @brief Create a restorer instance
 * @param[in] model_path path to image restoration model
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle handle of the created restorer, which must be destroyed
 * by \ref mmdeploy_restorer_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_restorer_create_by_path(const char* model_path, const char* device_name,
                                                  int device_id, mm_handle_t* handle);

/**
 * @brief Apply restorer to a batch of images
 * @param[in] handle restorer's handle created by \ref mmdeploy_restorer_create_by_path
 * @param[in] images a batch of images
 * @param[in] count number of images in the batch
 * @param[out] results a linear buffer contains the restored images, must be release
 * by \ref mmdeploy_restorer_release_result
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_restorer_apply(mm_handle_t handle, const mm_mat_t* images, int count,
                                         mm_mat_t** results);

/** @brief Release result buffer returned by \ref mmdeploy_restorer_apply
 * @param[in] results result buffer by restorer
 * @param[in] count length of \p result
 */
MMDEPLOY_API void mmdeploy_restorer_release_result(mm_mat_t* results, int count);

/**
 * @brief destroy restorer
 * @param[in] handle handle of restorer created by \ref mmdeploy_restorer_create_by_path
 */
MMDEPLOY_API void mmdeploy_restorer_destroy(mm_handle_t handle);

/******************************************************************************
 * Experimental asynchronous APIs */

MMDEPLOY_API int mmdeploy_restorer_create_v2(mm_model_t model, const char* device_name,
                                             int device_id, mmdeploy_exec_info_t exec_info,
                                             mm_handle_t* handle);

MMDEPLOY_API int mmdeploy_restorer_create_input(const mm_mat_t* mats, int mat_count,
                                                mmdeploy_value_t* value);

MMDEPLOY_API int mmdeploy_restorer_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                            mmdeploy_value_t* output);

MMDEPLOY_API int mmdeploy_restorer_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                               mmdeploy_sender_t* output);

MMDEPLOY_API int mmdeploy_restorer_get_result(mmdeploy_value_t output, mm_mat_t** results);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_SRC_APIS_C_RESTORER_H_
