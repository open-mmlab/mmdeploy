# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_transform_module)

add_subdirectory(transform)
add_subdirectory(cpu)
if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    add_subdirectory(cuda)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
build_target(${PROJECT_NAME} transform_module.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy::core)
add_library(mmdeploy::transform_module ALIAS ${PROJECT_NAME})
export_module(${PROJECT_NAME})
