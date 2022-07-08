# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmcls)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::mmcls ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} classifier CACHE INTERNAL "")
