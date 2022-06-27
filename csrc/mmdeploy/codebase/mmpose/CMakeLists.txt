# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmpose)


file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy::transform mmdeploy_opencv_utils)
add_library(mmdeploy::mmpose ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} pose_detector CACHE INTERNAL "")
