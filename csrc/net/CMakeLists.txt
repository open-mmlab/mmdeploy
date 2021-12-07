# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_net_module)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

if ("trt" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(trt)
endif ()

if ("pplnn" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(ppl)
endif ()

if ("ort" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(ort)
endif ()
if ("ncnn" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(ncnn)
endif ()

set_targets(${PROJECT_NAME} NET_MODULE_OBJ NET_MODULE_STATIC NET_MODULE_SHARED)
build_object_target(${NET_MODULE_OBJ} net_module.cpp)
target_link_libraries(${NET_MODULE_OBJ} PRIVATE mmdeploy::core::static)

build_static_target(${NET_MODULE_STATIC} ${NET_MODULE_OBJ} "PRIVATE")
add_library(mmdeploy::net_module::static ALIAS ${NET_MODULE_STATIC})

build_shared_target(${NET_MODULE_SHARED} ${NET_MODULE_OBJ} "PRIVATE")
add_library(mmdeploy::net_module ALIAS ${NET_MODULE_SHARED})

export_module(${NET_MODULE_STATIC} ${NET_MODULE_SHARED} ${NET_MODULE_OBJ})
