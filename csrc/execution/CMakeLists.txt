# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_execution)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
set(SRCS
        schedulers/timed_single_thread_context.cpp
        schedulers/schedulers.cpp)
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::execution ALIAS ${PROJECT_NAME})
