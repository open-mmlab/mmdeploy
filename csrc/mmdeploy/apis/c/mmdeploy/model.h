// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file model.h
 * @brief Interface to MMDeploy SDK Model
 */

#ifndef MMDEPLOY_SRC_APIS_C_MODEL_H_
#define MMDEPLOY_SRC_APIS_C_MODEL_H_

#include "mmdeploy/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_model* mmdeploy_model_t;

/**
 * @brief Create SDK Model instance from given model path
 * @param[in] path model path
 * @param[out] model sdk model instance that must be destroyed by \ref mmdeploy_model_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_model_create_by_path(const char* path, mmdeploy_model_t* model);

/**
 * @brief Create SDK Model instance from memory
 * @param[in] buffer a linear buffer contains the model information
 * @param[in] size size of \p buffer in bytes
 * @param[out] model sdk model instance that must be destroyed by \ref mmdeploy_model_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_model_create(const void* buffer, int size, mmdeploy_model_t* model);

/**
 * @brief Destroy model instance
 * @param[in] model sdk model instance created by \ref mmdeploy_model_create_by_path or \ref
 * mmdeploy_model_create
 */
MMDEPLOY_API void mmdeploy_model_destroy(mmdeploy_model_t model);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_SRC_APIS_C_MODEL_H_
