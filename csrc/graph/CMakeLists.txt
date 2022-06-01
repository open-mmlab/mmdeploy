# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_graph)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
set(SRCS
        common.cpp
        task.cpp
        pipeline.cpp
        inference.cpp
        flatten.cpp
        unflatten.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::graph ALIAS ${PROJECT_NAME})
