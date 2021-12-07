# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_transform_module)

add_subdirectory(transform)
add_subdirectory(cpu)
if ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES)
    add_subdirectory(cuda)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} TRANSFORM_MODULE_OBJ TRANSFORM_MODULE_STATIC TRANSFORM_MODULE_SHARED)

build_object_target(${TRANSFORM_MODULE_OBJ} transform_module.cpp)
target_link_libraries(${TRANSFORM_MODULE_OBJ}
        PRIVATE mmdeploy::core::static)

build_static_target(${TRANSFORM_MODULE_STATIC} ${TRANSFORM_MODULE_OBJ} "PRIVATE")
add_library(mmdeploy::transform_module::static ALIAS ${TRANSFORM_MODULE_STATIC})

build_shared_target(${TRANSFORM_MODULE_SHARED} ${TRANSFORM_MODULE_OBJ} "PRIVATE")
add_library(mmdeploy::transform_module ALIAS ${TRANSFORM_MODULE_SHARED})

export_module(${TRANSFORM_MODULE_STATIC} ${TRANSFORM_MODULE_SHARED} ${TRANSFORM_MODULE_OBJ})
