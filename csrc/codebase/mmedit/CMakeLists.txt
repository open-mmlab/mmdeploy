# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_mmedit)

include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} MMEDIT_OBJ MMEDIT_STATIC MMEDIT_SHARED)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
build_object_target(${MMEDIT_OBJ} "${SRCS}")
target_link_libraries(${MMEDIT_OBJ} PRIVATE
        mmdeploy::core::static opencv_core)

build_static_target(${MMEDIT_STATIC} ${MMEDIT_OBJ} "PRIVATE")
add_library(mmdeploy::mmedit::static ALIAS ${MMEDIT_STATIC})

build_shared_target(${MMEDIT_SHARED} ${MMEDIT_OBJ} "PRIVATE")
add_library(mmdeploy::mmedit ALIAS ${MMEDIT_SHARED})

export_module(${MMEDIT_STATIC} ${MMEDIT_SHARED} ${MMEDIT_OBJ})
