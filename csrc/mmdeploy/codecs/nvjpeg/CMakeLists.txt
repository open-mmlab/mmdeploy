# Copyright (c) OpenMMLab. All rights reserved.

if (NOT "cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    return()
endif ()

project(mmdeploy_nvjpeg)

set(SRCS jpeg_decoder.cpp)
mmdeploy_add_module(${PROJECT_NAME} LIBRARY "${SRCS}")
get_filename_component(_CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)

target_include_directories(${PROJECT_NAME} PRIVATE
            ${CUDA_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${_CUDA_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} nvjpeg)
add_library(mmdeploy::nvjpeg ALIAS ${PROJECT_NAME})

set(MMDEPLOY_CODECS ${MMDEPLOY_CODECS} jpeg_decoder CACHE INTERNAL "")
set(MMDEPLOY_CODECS_LIBS ${MMDEPLOY_CODECS_LIBS} mmdeploy_nvjpeg CACHE INTERNAL "")
