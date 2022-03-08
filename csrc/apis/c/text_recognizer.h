// Copyright (c) OpenMMLab. All rights reserved.

/**
 * @file text_recognizer.h
 * @brief Interface to MMOCR text recognition task
 */

#ifndef MMDEPLOY_SRC_APIS_C_TEXT_RECOGNIZER_H_
#define MMDEPLOY_SRC_APIS_C_TEXT_RECOGNIZER_H_

#include "common.h"
#include "text_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mm_text_recognize_t {
  char* text;
  float* score;
  int length;
} mm_text_recognize_t;

/**
 * @brief Create a text recognizer instance
 * @param[in] model an instance of mmocr text recognition model created by
 * \ref mmdeploy_model_create_by_path or \ref mmdeploy_model_create in \ref model.h
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle handle of the created text recognizer, which must be destroyed
 * by \ref mmdeploy_text_recognizer_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_text_recognizer_create(mm_model_t model, const char* device_name,
                                                 int device_id, mm_handle_t* handle);

/**
 * @brief Create a text recognizer instance
 * @param[in] model_path path to text recognition model
 * @param[in] device_name name of device, such as "cpu", "cuda", etc.
 * @param[in] device_id id of device.
 * @param[out] handle handle of the created text recognizer, which must be destroyed
 * by \ref mmdeploy_text_recognizer_destroy
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_text_recognizer_create_by_path(const char* model_path,
                                                         const char* device_name, int device_id,
                                                         mm_handle_t* handle);

/**
 * @brief Apply text recognizer to a batch of text images
 * @param[in] handle text recognizer's handle created by \ref
 * mmdeploy_text_recognizer_create_by_path
 * @param[in] images a batch of text images
 * @param[in] count number of images in the batch
 * @param[out] results a linear buffer contains the recognized text, must be release
 * by \ref mmdeploy_text_recognizer_release_result
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_text_recognizer_apply(mm_handle_t handle, const mm_mat_t* images,
                                                int count, mm_text_recognize_t** results);

/**
 * @brief Apply text recognizer to a batch of images supplied with text bboxes
 * @param[in] handle text recognizer's handle created by \ref
 * mmdeploy_text_recognizer_create_by_path
 * @param[in] images a batch of text images
 * @param[in] image_count number of images in the batch
 * @param[in] bboxes bounding boxes detected by text detector
 * @param[in] bbox_count number of bboxes of each \p images, must be same length as \p images
 * @param[out] results a linear buffer contains the recognized text, which has the same length as \p
 * bboxes, must be release by \ref mmdeploy_text_recognizer_release_result
 * @return status code of the operation
 */
MMDEPLOY_API int mmdeploy_text_recognizer_apply_bbox(mm_handle_t handle, const mm_mat_t* images,
                                                     int image_count,
                                                     const mm_text_detect_t* bboxes,
                                                     const int* bbox_count,
                                                     mm_text_recognize_t** results);

/** @brief Release result buffer returned by \ref mmdeploy_text_recognizer_apply or \ref
 * mmdeploy_text_recognizer_apply_bbox
 * @param[in] results result buffer by text recognizer
 * @param[in] count length of \p result
 */
MMDEPLOY_API void mmdeploy_text_recognizer_release_result(mm_text_recognize_t* results, int count);

/**
 * @brief destroy text recognizer
 * @param[in] handle handle of text recognizer created by \ref
 * mmdeploy_text_recognizer_create_by_path or \ref mmdeploy_text_recognizer_create
 */
MMDEPLOY_API void mmdeploy_text_recognizer_destroy(mm_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif  // MMDEPLOY_SRC_APIS_C_TEXT_RECOGNIZER_H_
