// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file image_decoder.h
 * @brief
 */

#ifndef MMDEPLOY_SRC_APIS_C_JPEG_H_
#define MMDEPLOY_SRC_APIS_C_JPEG_H_

#include "common.h"

/**
 * @brief handle of image decoder
 */
typedef struct mmdeploy_image_decoder* mmdeploy_image_decoder_t;

/**
 * @brief Create image decoder handle
 * @param[in] config config of decoder
 * @param[in] device_name name of device.
 * @param[in] device_id id of device
 * @param[out] decoder instance of a image decoder, which must be destroyed by \ref
 * mmdeploy_image_decoder_destroy
 */
MMDEPLOY_API int mmdeploy_image_decoder_create(mmdeploy_value_t config, const char* device_name,
                                               int device_id, mmdeploy_image_decoder_t* decoder);

/**
 * @brief Destroy image decoder handle
 * @param[in] decoder instance of a image decoder
 */
MMDEPLOY_API void mmdeploy_image_decoder_destroy(mmdeploy_image_decoder_t decoder);

/**
 * @brief Decode a batch of image image, not thread safe
 * @param[in] raw_data image buffer for each image
 * @param[in] length buffer length for each image
 * @param[in] count number of image
 * @param[in] format the output image format, only support bgr or rgb
 * @param[out] dev_results decode results with data on cuda
 * @param[out] host_results decode results with data on cpu
 */
MMDEPLOY_API int mmdeploy_image_decoder_apply(mmdeploy_image_decoder_t decoder,
                                              const char** raw_data, int* length, int count,
                                              mmdeploy_pixel_format_t format,
                                              mmdeploy_mat_t** dev_results,
                                              mmdeploy_mat_t** host_results);

/**
 * @brief Release the inference result buffer created by \ref mmdeploy_image_decoder_apply
 * @param[in] results decode results
 * @param[in] count the length of results
 */
MMDEPLOY_API void mmdeploy_image_decoder_release_result(mmdeploy_mat_t* results, int count);

#endif
