# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_cpu_transform_impl)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} CPU_TRANSFORM_OBJ CPU_TRANSFORM_STATIC CPU_TRANSFORM_SHARED)

set(SRCS
        collect_impl.cpp
        crop_impl.cpp
        image2tensor_impl.cpp
        load_impl.cpp
        normalize_impl.cpp
        opencv_utils.cpp
        opencv_utils.h
        pad_impl.cpp
        resize_impl.cpp)
build_object_target(${CPU_TRANSFORM_OBJ} "${SRCS}")
target_link_libraries(${CPU_TRANSFORM_OBJ} PUBLIC opencv_imgproc opencv_core)
target_link_libraries(${CPU_TRANSFORM_OBJ}
        PRIVATE mmdeploy::core::static)

build_static_target(${CPU_TRANSFORM_STATIC} ${CPU_TRANSFORM_OBJ} "PUBLIC")
add_library(mmdeploy::transform_impl::cpu::static ALIAS ${CPU_TRANSFORM_STATIC})

build_shared_target(${CPU_TRANSFORM_SHARED} ${CPU_TRANSFORM_OBJ} "PUBLIC")
add_library(mmdeploy::transform_impl::cpu ALIAS ${CPU_TRANSFORM_SHARED})

export_module(${CPU_TRANSFORM_STATIC} ${CPU_TRANSFORM_SHARED} ${CPU_TRANSFORM_OBJ})
