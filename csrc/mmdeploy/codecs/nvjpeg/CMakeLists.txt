# Copyright (c) OpenMMLab. All rights reserved.

if (NOT "cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    return()
endif ()

project(mmdeploy_nvjpeg_decoder)

set(SRCS nvjpeg_decoder.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
get_filename_component(_CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)

target_include_directories(${PROJECT_NAME} PRIVATE
            ${CUDA_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${_CUDA_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} nvjpeg)
