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

typedef struct mmdeploy_pipeline* mmdeploy_pipeline_t;

/**
 * Create pipeline
 * @param config
 * @param context
 * @param pipeline
 * @return
 */
MMDEPLOY_API int mmdeploy_pipeline_create_v3(mmdeploy_value_t config, mmdeploy_context_t context,
                                             mmdeploy_pipeline_t* pipeline);

/**
 * @brief Apply pipeline
 * @param[in] pipeline handle of the pipeline
 * @param[in] input input value
 * @param[out] output output value
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pipeline_apply(mmdeploy_pipeline_t pipeline, mmdeploy_value_t input,
                                         mmdeploy_value_t* output);

/**
 * Apply pipeline asynchronously
 * @param pipeline handle of the pipeline
 * @param input input sender that will be consumed by the operation
 * @param output output sender
 * @return status of the operation
 */
MMDEPLOY_API int mmdeploy_pipeline_apply_async(mmdeploy_pipeline_t pipeline,
                                               mmdeploy_sender_t input, mmdeploy_sender_t* output);

/**
 * @brief destroy pipeline
 * @param[in] pipeline
 */
MMDEPLOY_API void mmdeploy_pipeline_destroy(mmdeploy_pipeline_t pipeline);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_CSRC_APIS_C_PIPELINE_H_
