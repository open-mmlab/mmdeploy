# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_transform)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} TRANSFORM_OBJ TRANSFORM_STATIC TRANSFORM_SHARED)

set(SRCS
        collect.cpp
        compose.cpp
        crop.cpp
        image2tensor.cpp
        load.cpp
        normalize.cpp
        pad.cpp
        resize.cpp
        transform.cpp
        transform_utils.cpp)
build_object_target(${TRANSFORM_OBJ} "${SRCS}")
target_include_directories(${TRANSFORM_OBJ} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/preprocess>)
target_link_libraries(${TRANSFORM_OBJ} PRIVATE mmdeploy::core::static)

build_static_target(${TRANSFORM_STATIC} ${TRANSFORM_OBJ} "PUBLIC")
add_library(mmdeploy::transform::static ALIAS ${TRANSFORM_STATIC})

build_shared_target(${TRANSFORM_SHARED} ${TRANSFORM_OBJ} "PUBLIC")
add_library(mmdeploy::transform ALIAS ${TRANSFORM_SHARED})

export_module(${TRANSFORM_STATIC} ${TRANSFORM_SHARED} ${TRANSFORM_OBJ})
