# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ort_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
    add_library(${PROJECT_NAME} SHARED ort_net.cpp)
    target_include_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_DIR}/include)
    target_link_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_DIR}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy::core
            PUBLIC onnxruntime)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            -Wl,--whole-archive
            mmdeploy::onnxruntime::ops::static
            -Wl,--no-whole-archive)
    add_library(mmdeploy::ort_net ALIAS ${PROJECT_NAME})
    export_module(${PROJECT_NAME})
else ()
    message(ERROR "'ort_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
