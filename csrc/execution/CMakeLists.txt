# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_execution)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
set(SRCS
        execution.cpp
        schedulers/timed_single_thread_context.cpp
        graph/pipeline2.cpp
        graph/inference.cpp
        schedulers/schedulers.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::execution ALIAS ${PROJECT_NAME})
