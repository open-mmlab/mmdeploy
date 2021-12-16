# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmseg)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} MMSEG_OBJ MMSEG_STATIC MMSEG_SHARED)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
build_object_target(${MMSEG_OBJ} "${SRCS}")
target_link_libraries(${MMSEG_OBJ} PRIVATE
        mmdeploy::core::static opencv_core)

build_static_target(${MMSEG_STATIC} ${MMSEG_OBJ} "PRIVATE")
add_library(mmdeploy::mmseg::static ALIAS ${MMSEG_STATIC})

build_shared_target(${MMSEG_SHARED} ${MMSEG_OBJ} "PRIVATE")
add_library(mmdeploy::mmseg ALIAS ${MMSEG_SHARED})

export_module(${MMSEG_STATIC} ${MMSEG_SHARED} ${MMSEG_OBJ})
