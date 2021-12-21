# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cpu_device)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

file(GLOB_RECURSE SRCS "*.cpp")
build_target(${PROJECT_NAME} "${SRCS}")
target_link_libraries(${PROJECT_NAME} PUBLIC pthread PRIVATE mmdeploy::core)
add_library(mmdeploy::device::cpu ALIAS ${PROJECT_NAME})
export_module(${PROJECT_NAME})
