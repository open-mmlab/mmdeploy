# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cpu_device)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} CPU_DEVICE_OBJ CPU_DEVICE_STATIC CPU_DEVICE_SHARED)

file(GLOB_RECURSE SRCS "*.cpp")
build_object_target(${CPU_DEVICE_OBJ} "${SRCS}")
target_link_libraries(${CPU_DEVICE_OBJ} PRIVATE mmdeploy::core::static)

build_static_target(${CPU_DEVICE_STATIC} ${CPU_DEVICE_OBJ} "PRIVATE")
add_library(mmdeploy::device::cpu::static ALIAS ${CPU_DEVICE_STATIC})

build_shared_target(${CPU_DEVICE_SHARED} ${CPU_DEVICE_OBJ} "PRIVATE")
add_library(mmdeploy::device::cpu ALIAS ${CPU_DEVICE_SHARED})

export_module(${CPU_DEVICE_STATIC} ${CPU_DEVICE_SHARED} ${CPU_DEVICE_OBJ})
