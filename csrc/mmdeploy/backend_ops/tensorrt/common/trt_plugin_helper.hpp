// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_PLUGIN_HELPER_HPP
#define TRT_PLUGIN_HELPER_HPP
#include <cudnn.h>

#include <iostream>
#include <stdexcept>

#include "NvInferRuntime.h"

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype, cudnnDataType_t* cudnn_dtype);

// Enumerator for status
typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

#define ASSERT(assertion)                                                    \
  {                                                                          \
    if (!(assertion)) {                                                      \
      std::cerr << "#assertion" << __FILE__ << "," << __LINE__ << std::endl; \
      abort();                                                               \
    }                                                                        \
  }

#define CUASSERT(status_)                                                                       \
  {                                                                                             \
    auto s_ = status_;                                                                          \
    if (s_ != cudaSuccess) {                                                                    \
      std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << ", " << cudaGetErrorString(s_) \
                << std::endl;                                                                   \
    }                                                                                           \
  }
#define CUBLASASSERT(status_)                                               \
  {                                                                         \
    auto s_ = status_;                                                      \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << std::endl; \
    }                                                                       \
  }
#define CUERRORMSG(status_)                                                            \
  {                                                                                    \
    auto s_ = status_;                                                                 \
    if (s_ != 0) std::cerr << __FILE__ << ", " << __LINE__ << ", " << s_ << std::endl; \
  }

#ifndef DEBUG

#define CHECK(status)         \
  do {                        \
    if (status != 0) abort(); \
  } while (0)

#define ASSERT_PARAM(exp)                \
  do {                                   \
    if (!(exp)) return STATUS_BAD_PARAM; \
  } while (0)

#define ASSERT_FAILURE(exp)            \
  do {                                 \
    if (!(exp)) return STATUS_FAILURE; \
  } while (0)

#define CSC(call, err)               \
  do {                               \
    cudaError_t cudaStatus = call;   \
    if (cudaStatus != cudaSuccess) { \
      return err;                    \
    }                                \
  } while (0)

#define DEBUG_PRINTF(...) \
  do {                    \
  } while (0)

#else

#define ASSERT_PARAM(exp)                                                   \
  do {                                                                      \
    if (!(exp)) {                                                           \
      fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__); \
      return STATUS_BAD_PARAM;                                              \
    }                                                                       \
  } while (0)

#define ASSERT_FAILURE(exp)                                               \
  do {                                                                    \
    if (!(exp)) {                                                         \
      fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__); \
      return STATUS_FAILURE;                                              \
    }                                                                     \
  } while (0)

#define CSC(call, err)                                                                    \
  do {                                                                                    \
    cudaError_t cudaStatus = call;                                                        \
    if (cudaStatus != cudaSuccess) {                                                      \
      printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
      return err;                                                                         \
    }                                                                                     \
  } while (0)

#define CHECK(status)                                                                       \
  {                                                                                         \
    if (status != 0) {                                                                      \
      DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
      abort();                                                                              \
    }                                                                                       \
  }

#define DEBUG_PRINTF(...) \
  do {                    \
    printf(__VA_ARGS__);  \
  } while (0)

#endif

namespace mmdeploy {

const int MAXTENSORDIMS = 10;

struct TensorDesc {
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
  int dim;
};

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    // case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      throw std::runtime_error("Invalid DataType.");
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
  return size_t((origin_size + aligned_number - 1) / aligned_number) * aligned_number;
}

}  // namespace mmdeploy
#endif  // TRT_PLUGIN_HELPER_HPP
