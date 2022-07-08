# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmrotate)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy_opencv_utils)
add_library(mmdeploy::mmrotate ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} rotated_detector CACHE INTERNAL "")
