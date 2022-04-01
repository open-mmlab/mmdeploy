# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmpose)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy_opencv_utils)
add_library(mmdeploy::mmpose ALIAS ${PROJECT_NAME})
