# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmcls)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} MMCLS_OBJ MMCLS_STATIC MMCLS_SHARED)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
build_object_target(${MMCLS_OBJ} "${SRCS}")
target_link_libraries(${MMCLS_OBJ} PRIVATE
        mmdeploy::core::static)

build_static_target(${MMCLS_STATIC} ${MMCLS_OBJ} "PRIVATE")
add_library(mmdeploy::mmcls::static ALIAS ${MMCLS_STATIC})

build_shared_target(${MMCLS_SHARED} ${MMCLS_OBJ} "PRIVATE")
add_library(mmdeploy::mmcls ALIAS ${MMCLS_SHARED})

export_module(${MMCLS_STATIC} ${MMCLS_SHARED} ${MMCLS_OBJ} ${MMCLS_OBJ})
