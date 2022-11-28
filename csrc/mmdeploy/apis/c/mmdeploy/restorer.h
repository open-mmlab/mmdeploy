// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file restorer.h
 * @brief Interface to MMEditing image restoration task
 */

#ifndef MMDEPLOY_SRC_APIS_C_RESTORER_H_
#define MMDEPLOY_SRC_APIS_C_RESTORER_H_

#include "common.h"
#include "executor.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_restorer* mmdeploy_restorer_t;

/**
 * @brief Create a restorer instance
 * @param[in] model an instance of image restoration model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] restorer handle of the created restorer, which must be destroyed
 * by \ref mmdeploy_restorer_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_restorer_create(mmdeploy_model_t model, const char* device_name,
                                          int device_id, mmdeploy_restorer_t* restorer);

/**
 * @brief Create a restorer instance
 * @param[in] model_path path to image restoration model
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] restorer handle of the created restorer, which must be destroyed
 * by \ref mmdeploy_restorer_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_restorer_create_by_path(const char* model_path, const char* device_name,
                                                  int device_id, mmdeploy_restorer_t* restorer);

/**
 * @brief Apply restorer to a batch of images
 * @param[in] restorer restorer's handle created by \ref mmdeploy_restorer_create_by_path
 * @param[in] images a batch of images
 * @param[in] count number of images in the batch
 * @param[out] results a linear buffer contains the restored images, must be release
 * by \ref mmdeploy_restorer_release_result
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_restorer_apply(mmdeploy_restorer_t restorer, const mmdeploy_mat_t* images,
                                         int count, mmdeploy_mat_t** results);

/** @brief Release result buffer returned by \ref mmdeploy_restorer_apply
 * @param[in] results result buffer by restorer
 * @param[in] count length of \p result
 */
MMDEPLOY_API void mmdeploy_restorer_release_result(mmdeploy_mat_t* results, int count);

/**
 * @brief destroy restorer
 * @param[in] restorer handle of restorer created by \ref mmdeploy_restorer_create_by_path
 */
MMDEPLOY_API void mmdeploy_restorer_destroy(mmdeploy_restorer_t restorer);

/******************************************************************************
 * Experimental asynchronous APIs */

MMDEPLOY_API int mmdeploy_restorer_create_v2(mmdeploy_model_t model, mmdeploy_context_t context,
                                             mmdeploy_restorer_t* restorer);

MMDEPLOY_API int mmdeploy_restorer_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                                mmdeploy_value_t* value);

MMDEPLOY_API int mmdeploy_restorer_apply_v2(mmdeploy_restorer_t restorer, mmdeploy_value_t input,
                                            mmdeploy_value_t* output);

MMDEPLOY_API int mmdeploy_restorer_apply_async(mmdeploy_restorer_t restorer,
                                               mmdeploy_sender_t input, mmdeploy_sender_t* output);

MMDEPLOY_API int mmdeploy_restorer_get_result(mmdeploy_value_t output, mmdeploy_mat_t** results);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_SRC_APIS_C_RESTORER_H_
