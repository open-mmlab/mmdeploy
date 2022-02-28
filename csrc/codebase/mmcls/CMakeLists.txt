# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmcls)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_library(mmdeploy::mmcls ALIAS ${PROJECT_NAME})
