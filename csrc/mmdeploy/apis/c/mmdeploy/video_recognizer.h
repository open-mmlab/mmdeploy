// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file video_recognizer.h
 * @brief Interface to MMACTION video recognition task
 */

#ifndef MMDEPLOY_VIDEO_RECOGNIZER_H
#define MMDEPLOY_VIDEO_RECOGNIZER_H

#include "mmdeploy/common.h"
#include "mmdeploy/executor.h"
#include "mmdeploy/model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_video_recognition_t {
  int label_id;
  float score;
} mmdeploy_video_recognition_t;

typedef struct mmdeploy_video_sample_info_t {
  int clip_len;
  int num_clips;
} mmdeploy_video_sample_info_t;

typedef struct mmdeploy_video_recognizer* mmdeploy_video_recognizer_t;

/**
 * @brief Create video recognizer's handle
 * @param[in] model an instance of mmaction sdk model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] recognizer handle of the created video recognizer, which must be destroyed
 * by \ref mmdeploy_video_recognizer_destroy
 * @return status of creating video recognizer's handle
 */
MMDEPLOY_API int mmdeploy_video_recognizer_create(mmdeploy_model_t model, const char* device_name,
                                                  int device_id,
                                                  mmdeploy_video_recognizer_t* recognizer);

/**
 * @brief Create a video recognizer instance
 * @param[in] model_path path to video recognition model
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] recognizer handle of the created video recognizer, which must be destroyed
 * by \ref mmdeploy_video_recognizer_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_video_recognizer_create_by_path(const char* model_path,
                                                          const char* device_name, int device_id,
                                                          mmdeploy_video_recognizer_t* recognizer);

/**
 * @brief Apply video recognizer to a batch of videos
 * @param[in] recognizer video recognizer's handle created by \ref
 * mmdeploy_video_recognizer_create_by_path
 * @param[in] images a batch of videos
 * @param[in] video_info video information of each video
 * @param[in] video_count number of videos
 * @param[out] results a linear buffer contains the recognized video, must be release
 * by \ref mmdeploy_video_recognizer_release_result
 * @param[out] result_count a linear buffer with length being \p video_count to save the number of
 * recognition results of each video. It must be released by \ref
 * mmdeploy_video_recognizer_release_result
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_video_recognizer_apply(mmdeploy_video_recognizer_t recognizer,
                                                 const mmdeploy_mat_t* images,
                                                 const mmdeploy_video_sample_info_t* video_info,
                                                 int video_count,
                                                 mmdeploy_video_recognition_t** results,
                                                 int** result_count);

/** @brief Release result buffer returned by \ref mmdeploy_video_recognizer_apply
 * @param[in] results result buffer by video recognizer
 * @param[in] result_count \p results size buffer
 * @param[in] video_count length of \p result_count
 */
MMDEPLOY_API void mmdeploy_video_recognizer_release_result(mmdeploy_video_recognition_t* results,
                                                           int* result_count, int video_count);

/**
 * @brief destroy video recognizer
 * @param[in] recognizer handle of video recognizer created by \ref
 * mmdeploy_video_recognizer_create_by_path or \ref mmdeploy_video_recognizer_create
 */
MMDEPLOY_API void mmdeploy_video_recognizer_destroy(mmdeploy_video_recognizer_t recognizer);

/**
 * @brief Same as \ref mmdeploy_video_recognizer_create, but allows to control execution context of
 * tasks via context
 */
MMDEPLOY_API int mmdeploy_video_recognizer_create_v2(mmdeploy_model_t model,
                                                     mmdeploy_context_t context,
                                                     mmdeploy_video_recognizer_t* recognizer);

/**
 * @brief Pack video recognizer inputs into mmdeploy_value_t
 * @param[in] images a batch of videos
 * @param[in] video_info video information of each video
 * @param[in] video_count number of videos in the batch
 * @param[out] value created value
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_video_recognizer_create_input(
    const mmdeploy_mat_t* images, const mmdeploy_video_sample_info_t* video_info, int video_count,
    mmdeploy_value_t* value);

/**
 * @brief Apply video recognizer to a batch of videos
 * @param[in] input packed input
 * @param[out] output inference output
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_video_recognizer_apply_v2(mmdeploy_video_recognizer_t recognizer,
                                                    mmdeploy_value_t input,
                                                    mmdeploy_value_t* output);

/**
 * @brief Apply video recognizer to a batch of videos
 * @param[in] output inference output
 * @param[out] results structured output
 * @param[out] result_count number of each videos
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_video_recognizer_get_result(mmdeploy_value_t output,
                                                      mmdeploy_video_recognition_t** results,
                                                      int** result_count);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_VIDEO_RECOGNIZER_H
