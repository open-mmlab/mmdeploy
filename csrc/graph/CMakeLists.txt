# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_graph)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set(SRCS
        common.cpp
        inference.cpp
        pipeline.cpp
        task.cpp
        flatten.cpp
        unflatten.cpp)
build_target(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy::core)
add_library(mmdeploy::graph ALIAS ${PROJECT_NAME})
export_module(${PROJECT_NAME})
