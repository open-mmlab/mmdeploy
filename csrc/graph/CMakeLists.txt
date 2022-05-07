# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_graph)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
set(SRCS
        common.cpp
        inference.cpp
        pipeline.cpp
        task.cpp
        flatten.cpp
        unflatten.cpp
        async/pipeline.cpp
        async/inference.cpp
        async/flatten.cpp
        async/unflatten.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::graph ALIAS ${PROJECT_NAME})
