# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_opencv_utils)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

mmdeploy_add_library(${PROJECT_NAME} opencv_utils.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE mmdeploy::core
        PUBLIC ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME}
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

#export_module(${PROJECT_NAME})
