// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file video_decoder.h
 * @brief
 */

#ifndef MMDEPLOY_SRC_APIS_VIDEO_DECODER_H_
#define MMDEPLOY_SRC_APIS_VIDEO_DECODER_H_

#include "common.h"

/**
 * @brief handle of video decoder
 */
typedef struct mmdeploy_video_decoder* mmdeploy_video_decoder_t;

/**
 * @brief Params of video decoder
 */
typedef struct mmdeploy_video_decoder_params_t {
  const char* path;
  mmdeploy_pixel_format_t format;
} mmdeploy_video_decoder_params_t;

/**
 * @brief video info
 */
typedef struct mmdeploy_video_info_t {
  int width;
  int height;
  int fourcc;
  double fps;
} mmdeploy_video_info_t;

/**
 * @brief Create video decoder handle
 * @param[in] params config of decoder
 * @param[in] device_name name of device.
 * @param[in] device_id id of device
 * @param[out] decoder instance of a video decoder, which must be destroyed by \ref
 * mmdeploy_video_decoder_destroy
 * @return status of decoder's handle
 */
MMDEPLOY_API int mmdeploy_video_decoder_create(mmdeploy_video_decoder_params_t params,
                                               const char* device_name, int device_id,
                                               mmdeploy_video_decoder_t* decoder);

/**
 * @brief Get video information
 * @param[in] decoder decoder handle
 * @param[out] video information
 */
MMDEPLOY_API int mmdeploy_video_decoder_info(mmdeploy_video_decoder_t decoder,
                                             mmdeploy_video_info_t* info);

/**
 * @brief Destroy video decoder handle
 * @param[in] decoder instance of a video decoder
 */
MMDEPLOY_API void mmdeploy_video_decoder_destroy(mmdeploy_video_decoder_t decoder);

/**
 * @brief Decode a frame, not thread safe
 * @param[in] decoder decoder handle
 * @param[out] dev_results decode results with data on cuda
 * @param[out] host_results decode results with data on cpu
 * @return status of decoder's handle
 */
MMDEPLOY_API int mmdeploy_video_decoder_read(mmdeploy_video_decoder_t decoder,
                                             mmdeploy_mat_t** dev_results,
                                             mmdeploy_mat_t** host_results);

/**
 *  @brief Grab a frame
 * @param[in] decoder decoder handle
 * @return status of decoder's handle
 */
MMDEPLOY_API int mmdeploy_video_decoder_grab(mmdeploy_video_decoder_t decoder);

/**
 *  @brief Retrieve last grabed frame
 *  @param[in] decoder decoder handle
 *  @param[out] dev_results decode results with data on cuda
 *  @param[out] host_results decode results with data on cpu
 *  @return status of decoder's handle
 */
MMDEPLOY_API int mmdeploy_video_decoder_retrieve(mmdeploy_video_decoder_t decoder,
                                                 mmdeploy_mat_t** dev_results,
                                                 mmdeploy_mat_t** host_results);

/**
 * @brief Release the inference result buffer created by \ref mmdeploy_video_decoder_apply
 * @param[in] results decode results
 * @param[in] count the length of results
 */
MMDEPLOY_API void mmdeploy_video_decoder_release_result(mmdeploy_mat_t* results, int count);

#endif  // MMDEPLOY_SRC_APIS_VIDEO_DECODER_H_
