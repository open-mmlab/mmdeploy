# Copyright (c) OpenMMLab. All rights reserved.

include(${CMAKE_SOURCE_DIR}/cmake/modules/FindTENSORRT.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/modules/FindCUDNN.cmake)
find_path(
        TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
if (TENSORRT_INCLUDE_DIR)
    message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
else ()
    message(ERROR "Cannot find TensorRT headers")
endif ()

find_library(
        TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(
        TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER}
        ${TENSORRT_LIBRARY_INFER_PLUGIN})
if (TENSORRT_LIBRARY_INFER
        AND TENSORRT_LIBRARY_INFER_PLUGIN)
    message(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")
else ()
    message(FATAL_ERROR "Cannot find TensorRT libs")
endif ()

find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR
        TENSORRT_LIBRARY)
if (NOT TENSORRT_FOUND)
    message(ERROR "Cannot find TensorRT library.")
endif ()
