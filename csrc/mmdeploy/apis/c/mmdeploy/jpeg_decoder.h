// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file jpeg_decoder.h
 * @brief
 */

#ifndef MMDEPLOY_SRC_APIS_C_JPEG_H_
#define MMDEPLOY_SRC_APIS_C_JPEG_H_

#include "common.h"

/**
 * @brief handle of jpeg decoder
 */
typedef struct mmdeploy_jpeg_decoder* mmdeploy_jpeg_decoder_t;

/**
 * @brief Create jpeg decoder handle
 * @param[in] device_id id of cuda device.
 * @param[out] decoder instance of a jpeg decoder, which must be destroyed by \ref
 * mmdeploy_jpeg_decoder_destroy
 */
MMDEPLOY_API int mmdeploy_jpeg_decoder_create(int device_id, mmdeploy_jpeg_decoder_t* decoder);

/**
 * @brief Destroy jpeg decoder handle
 * @param[in] decoder instance of a jpeg decoder
 */
MMDEPLOY_API void mmdeploy_jpeg_decoder_destroy(mmdeploy_jpeg_decoder_t decoder);

/**
 * @brief Decode a batch of jpeg image, not thread safe
 * @param[in] raw_data image buffer for each image
 * @param[in] length buffer length for each image
 * @param[in] count number of image
 * @param[in] format the output image format, only support bgr or rgb
 * @param[out] dev_results decode results with data on cuda
 * @param[out] host_results decode results with data on cpu
 */
MMDEPLOY_API int mmdeploy_jpeg_decoder_apply(mmdeploy_jpeg_decoder_t decoder, const char** raw_data,
                                             int* length, int count, mmdeploy_pixel_format_t format,
                                             mmdeploy_mat_t** dev_results,
                                             mmdeploy_mat_t** host_results);

/**
 * @brief Release the inference result buffer created by \ref mmdeploy_jpeg_decoder_apply
 * @param[in] results decode results
 * @param[in] count the length of results
 */
MMDEPLOY_API void mmdeploy_jpeg_decoder_release_result(mmdeploy_mat_t* results, int count);

#endif
