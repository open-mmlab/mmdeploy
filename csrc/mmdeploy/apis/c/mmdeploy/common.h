// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_COMMON_H
#define MMDEPLOY_COMMON_H

#include <stdint.h>  // NOLINT

#ifndef MMDEPLOY_EXPORT
#ifdef _MSC_VER
#define MMDEPLOY_EXPORT __declspec(dllexport)
#else
#define MMDEPLOY_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifndef MMDEPLOY_API
#ifdef MMDEPLOY_API_EXPORTS
#define MMDEPLOY_API MMDEPLOY_EXPORT
#else
#define MMDEPLOY_API
#endif
#endif

// clang-format off

typedef enum mmdeploy_pixel_format_t{
  MMDEPLOY_PIXEL_FORMAT_BGR,
  MMDEPLOY_PIXEL_FORMAT_RGB,
  MMDEPLOY_PIXEL_FORMAT_GRAYSCALE,
  MMDEPLOY_PIXEL_FORMAT_NV12,
  MMDEPLOY_PIXEL_FORMAT_NV21,
  MMDEPLOY_PIXEL_FORMAT_BGRA,
  MMDEPLOY_PIXEL_FORMAT_COUNT
} mmdeploy_pixel_format_t;

typedef enum mmdeploy_data_type_t{
  MMDEPLOY_DATA_TYPE_FLOAT,
  MMDEPLOY_DATA_TYPE_HALF,
  MMDEPLOY_DATA_TYPE_UINT8,
  MMDEPLOY_DATA_TYPE_INT32,
  MMDEPLOY_DATA_TYPE_COUNT
} mmdeploy_data_type_t;

typedef enum mmdeploy_status_t {
  MMDEPLOY_SUCCESS          = 0,
  MMDEPLOY_E_INVALID_ARG    = 1,
  MMDEPLOY_E_NOT_SUPPORTED  = 2,
  MMDEPLOY_E_OUT_OF_RANGE   = 3,
  MMDEPLOY_E_OUT_OF_MEMORY  = 4,
  MMDEPLOY_E_FILE_NOT_EXIST = 5,
  MMDEPLOY_E_FAIL           = 6,
  MMDEPLOY_STATUS_COUNT     = 7
} mmdeploy_status_t;

// clang-format on

typedef struct mmdeploy_device* mmdeploy_device_t;

typedef struct mmdeploy_mat_t {
  uint8_t* data;
  int height;
  int width;
  int channel;
  mmdeploy_pixel_format_t format;
  mmdeploy_data_type_t type;
  mmdeploy_device_t device;
} mmdeploy_mat_t;

typedef struct mmdeploy_rect_t {
  float left;
  float top;
  float right;
  float bottom;
} mmdeploy_rect_t;

typedef struct mmdeploy_point_t {
  float x;
  float y;
} mmdeploy_point_t;

typedef struct mmdeploy_value* mmdeploy_value_t;

typedef struct mmdeploy_context* mmdeploy_context_t;

typedef enum mmdeploy_context_type_t {
  MMDEPLOY_TYPE_DEVICE = 0,
  MMDEPLOY_TYPE_STREAM = 1,
  MMDEPLOY_TYPE_MODEL = 2,
  MMDEPLOY_TYPE_SCHEDULER = 3,
  MMDEPLOY_TYPE_MAT = 4,
} mmdeploy_context_type_t;

#if __cplusplus
extern "C" {
#endif

/**
 * Copy value
 * @param value
 * @return
 */
MMDEPLOY_API mmdeploy_value_t mmdeploy_value_copy(mmdeploy_value_t value);

/**
 * Destroy value
 * @param value
 */
MMDEPLOY_API void mmdeploy_value_destroy(mmdeploy_value_t value);

/**
 * Create device handle
 * @param device_name
 * @param device_id
 * @param device
 * @return
 */
MMDEPLOY_API int mmdeploy_device_create(const char* device_name, int device_id,
                                        mmdeploy_device_t* device);

/**
 * Destroy device handle
 * @param device
 */
MMDEPLOY_API void mmdeploy_device_destroy(mmdeploy_device_t device);

/**
 * Create context
 * @param context
 * @return
 */
MMDEPLOY_API int mmdeploy_context_create(mmdeploy_context_t* context);

/**
 * Create context
 * @param device_name
 * @param device_id
 * @param context
 * @return
 */
MMDEPLOY_API int mmdeploy_context_create_by_device(const char* device_name, int device_id,
                                                   mmdeploy_context_t* context);

/**
 * Destroy context
 * @param context
 */
MMDEPLOY_API void mmdeploy_context_destroy(mmdeploy_context_t context);

/**
 * Add context object
 * @param context
 * @param type
 * @param name
 * @param object
 * @return
 */
MMDEPLOY_API int mmdeploy_context_add(mmdeploy_context_t context, mmdeploy_context_type_t type,
                                      const char* name, const void* object);

/**
 * Create input value from array of mats
 * @param mats
 * @param mat_count
 * @param value
 * @return
 */
MMDEPLOY_API int mmdeploy_common_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                              mmdeploy_value_t* value);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_COMMON_H
