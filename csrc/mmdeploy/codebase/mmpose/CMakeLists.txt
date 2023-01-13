# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmpose)

find_package(OpenCV REQUIRED)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} MMPOSE_SRCS)

mmdeploy_add_module(${PROJECT_NAME} ${MMPOSE_SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy::transform
        mmdeploy_operation
        ${OpenCV_LIBS})
add_library(mmdeploy::mmpose ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} pose_detector CACHE INTERNAL "")
