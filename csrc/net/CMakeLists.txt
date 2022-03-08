# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_net_module)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

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

if ("openvino" IN_LIST MMDEPLOY_TARGET_BACKENDS)
    add_subdirectory(openvino)
endif ()

mmdeploy_add_module(${PROJECT_NAME} net_module.cpp)
add_library(mmdeploy::net_module ALIAS ${PROJECT_NAME})
