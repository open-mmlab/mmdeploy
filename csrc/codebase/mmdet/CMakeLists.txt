# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmdet)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} MMDET_OBJ MMDET_STATIC MMDET_SHARED)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
build_object_target(${MMDET_OBJ} "${SRCS}")
target_link_libraries(${MMDET_OBJ} PRIVATE
        mmdeploy::core::static opencv_core)

build_static_target(${MMDET_STATIC} ${MMDET_OBJ} "PRIVATE")
add_library(mmdeploy::mmdet::static ALIAS ${MMDET_STATIC})

build_shared_target(${MMDET_SHARED} ${MMDET_OBJ} "PRIVATE")
add_library(mmdeploy::mmdet ALIAS ${MMDET_SHARED})

export_module(${MMDET_STATIC} ${MMDET_SHARED} ${MMDET_OBJ})
