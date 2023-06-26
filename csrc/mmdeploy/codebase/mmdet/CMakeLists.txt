# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmdet)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy_opencv_utils mmdeploy_operation)

add_library(mmdeploy::mmdet ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} detector  CACHE INTERNAL "")
