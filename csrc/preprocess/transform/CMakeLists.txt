# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_transform)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

set(SRCS
    collect.cpp
    compose.cpp
    crop.cpp
    image2tensor.cpp
    load.cpp
    normalize.cpp
    pad.cpp
    resize.cpp
    transform.cpp)
build_target(${PROJECT_NAME} "${SRCS}")
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/preprocess>)
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy::core)
add_library(mmdeploy::transform ALIAS ${PROJECT_NAME})
export_module(${PROJECT_NAME})
