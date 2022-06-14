// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_APIS_C_PIPELINE_H_
#define MMDEPLOY_CSRC_APIS_C_PIPELINE_H_

#include "common.h"
#include "executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Experimental pipeline APIs */

/**
 * @brief Create pipeline
 * @param[in] config config of the pipeline
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[in] exec_info execution options
 * @param[out] handle handle of the pipeline
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pipeline_create(mmdeploy_value_t config, const char* device_name,
                                          int device_id, mmdeploy_exec_info_t exec_info,
                                          mm_handle_t* handle);

/**
 * @brief Apply pipeline
 * @param[in] handle handle of the pipeline
 * @param[in] input input value
 * @param[out] output output value
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pipeline_apply(mm_handle_t handle, mmdeploy_value_t input,
                                         mmdeploy_value_t* output);

/**
 * Apply pipeline asynchronously
 * @param handle handle of the pipeline
 * @param input input sender that will be consumed by the operation
 * @param output output sender
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pipeline_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                               mmdeploy_sender_t* output);

/**
 * @brief destroy pipeline
 * @param[in] handle
 */
MMDEPLOY_API void mmdeploy_pipeline_destroy(mm_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_CSRC_APIS_C_PIPELINE_H_
