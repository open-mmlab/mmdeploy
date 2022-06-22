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

typedef struct mmdeploy_mat_t {
  uint8_t* data;
  int height;
  int width;
  int channel;
  mmdeploy_pixel_format_t format;
  mmdeploy_data_type_t type;
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

#if __cplusplus
extern "C" {
#endif

MMDEPLOY_API mmdeploy_value_t mmdeploy_value_copy(mmdeploy_value_t value);

MMDEPLOY_API int mmdeploy_value_destroy(mmdeploy_value_t value);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_COMMON_H
