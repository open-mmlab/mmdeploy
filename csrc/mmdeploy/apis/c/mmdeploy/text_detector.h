// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file text_detector.h
 * @brief Interface to MMOCR text detection task
 */

#ifndef MMDEPLOY_TEXT_DETECTOR_H
#define MMDEPLOY_TEXT_DETECTOR_H

#include "common.h"
#include "executor.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmdeploy_text_detection_t {
  mmdeploy_point_t bbox[4];  ///< a text bounding box of which the vertex are in clock-wise
  float score;
} mmdeploy_text_detection_t;

typedef struct mmdeploy_text_detector* mmdeploy_text_detector_t;

/**
 * @brief Create text-detector's handle
 * @param[in] model an instance of mmocr text detection model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] detector instance of a text-detector, which must be destroyed
 * by \ref mmdeploy_text_detector_destroy
 * @return status of creating text-detector's handle
 */
MMDEPLOY_API int mmdeploy_text_detector_create(mmdeploy_model_t model, const char* device_name,
                                               int device_id, mmdeploy_text_detector_t* detector);

/**
 * @brief Create text-detector's handle
 * @param[in] model_path path to text detection model
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device
 * @param[out] detector instance of a text-detector, which must be destroyed
 * by \ref mmdeploy_text_detector_destroy
 * @return status of creating text-detector's handle
 */
MMDEPLOY_API int mmdeploy_text_detector_create_by_path(const char* model_path,
                                                       const char* device_name, int device_id,
                                                       mmdeploy_text_detector_t* detector);

/**
 * @brief Apply text-detector to batch images and get their inference results
 * @param[in] detector text-detector's handle created by \ref mmdeploy_text_detector_create_by_path
 * @param[in] mats a batch of images
 * @param[in] mat_count number of images in the batch
 * @param[out] results a linear buffer to save text detection results of each
 * image. It must be released by calling \ref mmdeploy_text_detector_release_result
 * @param[out] result_count a linear buffer of length \p mat_count to save the number of detection
 * results of each image. It must be released by \ref mmdeploy_detector_release_result
 * @return status of inference
 */
MMDEPLOY_API int mmdeploy_text_detector_apply(mmdeploy_text_detector_t detector,
                                              const mmdeploy_mat_t* mats, int mat_count,
                                              mmdeploy_text_detection_t** results,
                                              int** result_count);

/** @brief Release the inference result buffer returned by \ref mmdeploy_text_detector_apply
 * @param[in] results text detection result buffer
 * @param[in] result_count  \p results size buffer
 * @param[in] count the length of buffer \p result_count
 */
MMDEPLOY_API void mmdeploy_text_detector_release_result(mmdeploy_text_detection_t* results,
                                                        const int* result_count, int count);

/**
 * @brief Destroy text-detector's handle
 * @param[in] detector text-detector's handle created by \ref mmdeploy_text_detector_create_by_path
 * or \ref mmdeploy_text_detector_create
 */
MMDEPLOY_API void mmdeploy_text_detector_destroy(mmdeploy_text_detector_t detector);

/******************************************************************************
 * Experimental asynchronous APIs */

/**
 * @brief Same as \ref mmdeploy_text_detector_create, but allows to control execution context of
 * tasks via context
 */
MMDEPLOY_API int mmdeploy_text_detector_create_v2(mmdeploy_model_t model,
                                                  mmdeploy_context_t context,
                                                  mmdeploy_text_detector_t* detector);

/**
 * @brief Pack text-detector inputs into mmdeploy_value_t
 * @param[in] mats a batch of images
 * @param[in] mat_count number of images in the batch
 * @return the created value
 */
MMDEPLOY_API int mmdeploy_text_detector_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                                     mmdeploy_value_t* input);

/**
 * @brief Same as \ref mmdeploy_text_detector_apply, but input and output are packed in \ref
 * mmdeploy_value_t.
 */
MMDEPLOY_API int mmdeploy_text_detector_apply_v2(mmdeploy_text_detector_t detector,
                                                 mmdeploy_value_t input, mmdeploy_value_t* output);

/**
 * @brief Apply text-detector asynchronously
 * @param[in] detector handle to the detector
 * @param[in] input input sender that will be consumed by the operation
 * @return output sender
 */
MMDEPLOY_API int mmdeploy_text_detector_apply_async(mmdeploy_text_detector_t detector,
                                                    mmdeploy_sender_t input,
                                                    mmdeploy_sender_t* output);

/**
 * @brief Unpack detector output from a mmdeploy_value_t
 * @param[in] output output sender returned by applying a detector
 * @param[out] results a linear buffer to save detection results of each image. It must be
 * released by \ref mmdeploy_text_detector_release_result
 * @param[out] result_count a linear buffer with length number of input images to save the
 * number of detection results of each image. Must be released by \ref
 * mmdeploy_text_detector_release_result
 * @return status of the operation
 */
MMDEPLOY_API
int mmdeploy_text_detector_get_result(mmdeploy_value_t output, mmdeploy_text_detection_t** results,
                                      int** result_count);

typedef int (*mmdeploy_text_detector_continue_t)(mmdeploy_text_detection_t* results,
                                                 int* result_count, void* context,
                                                 mmdeploy_sender_t* output);

// MMDEPLOY_API int mmdeploy_text_detector_apply_async_v2(mm_handle_t handle, const mm_mat_t* imgs,
//                                                        int img_count,
//                                                        mmdeploy_text_detector_continuation_t
//                                                        cont, void* context, mmdeploy_sender_t*
//                                                        output);

MMDEPLOY_API int mmdeploy_text_detector_apply_async_v3(mmdeploy_text_detector_t detector,
                                                       const mmdeploy_mat_t* imgs, int img_count,
                                                       mmdeploy_sender_t* output);

MMDEPLOY_API int mmdeploy_text_detector_continue_async(mmdeploy_sender_t input,
                                                       mmdeploy_text_detector_continue_t cont,
                                                       void* context, mmdeploy_sender_t* output);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_TEXT_DETECTOR_H
