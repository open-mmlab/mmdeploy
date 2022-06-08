// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_COMMON_H
#define MMDEPLOY_COMMON_H

#include <stdint.h>

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

typedef enum {
  MM_BGR,
  MM_RGB,
  MM_GRAYSCALE,
  MM_NV12,
  MM_NV21,
  MM_BGRA,
  MM_UNKNOWN_PIXEL_FORMAT
} mm_pixel_format_t;

typedef enum {
  MM_FLOAT,
  MM_HALF,
  MM_INT8,
  MM_INT32,
  MM_UNKNOWN_DATA_TYPE
} mm_data_type_t;

enum mm_status_t {
  MM_SUCCESS          = 0,
  MM_E_INVALID_ARG    = 1,
  MM_E_NOT_SUPPORTED  = 2,
  MM_E_OUT_OF_RANGE   = 3,
  MM_E_OUT_OF_MEMORY  = 4,
  MM_E_FILE_NOT_EXIST = 5,
  MM_E_FAIL           = 6,
  MM_E_UNKNOWN        = -1,
};

// clang-format on

typedef void* mm_handle_t;

typedef void* mm_model_t;

typedef struct mm_mat_t {
  uint8_t* data;
  int height;
  int width;
  int channel;
  mm_pixel_format_t format;
  mm_data_type_t type;
} mm_mat_t;

typedef struct mm_rect_t {
  float left;
  float top;
  float right;
  float bottom;
} mm_rect_t;

typedef struct mm_pointi_t {
  int x;
  int y;
} mm_pointi_t;

typedef struct mm_pointf_t {
  float x;
  float y;
} mm_pointf_t;

typedef struct mmdeploy_value* mmdeploy_value_t;

#if __cplusplus
extern "C" {
#endif

MMDEPLOY_API mmdeploy_value_t mmdeploy_value_copy(mmdeploy_value_t input);

MMDEPLOY_API int mmdeploy_value_destroy(mmdeploy_value_t value);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_COMMON_H
