# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmpose)

find_package(OpenCV REQUIRED)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} MMPOSE_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/pose_tracker POSE_TRACKER_SRCS)

mmdeploy_add_module(${PROJECT_NAME} ${MMPOSE_SRCS} ${POSE_TRACKER_SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy::transform
        mmdeploy_operation
        mmdeploy_opencv_utils
        ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../apis/c)
add_library(mmdeploy::mmpose ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} pose_detector pose_tracker CACHE INTERNAL "")
