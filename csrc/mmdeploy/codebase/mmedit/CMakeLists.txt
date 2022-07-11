# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmedit)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core)
add_library(mmdeploy::mmedit ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} restorer CACHE INTERNAL "")
