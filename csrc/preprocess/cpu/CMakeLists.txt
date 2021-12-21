# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cpu_transform_impl)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

set(SRCS
        collect_impl.cpp
        crop_impl.cpp
        image2tensor_impl.cpp
        load_impl.cpp
        normalize_impl.cpp
        opencv_utils.cpp
        opencv_utils.h
        pad_impl.cpp
        resize_impl.cpp)
build_target(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME}
        PUBLIC opencv_imgproc opencv_core
        PRIVATE mmdeploy::core)
add_library(mmdeploy::transform_impl::cpu ALIAS ${PROJECT_NAME})
export_module(${PROJECT_NAME})
