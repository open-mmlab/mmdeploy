# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_openvino_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
    find_package(InferenceEngine REQUIRED)

    add_library(${PROJECT_NAME} SHARED openvino_net.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            mmdeploy::core
            ${InferenceEngine_LIBRARIES})
    add_library(mmdeploy::openvino_net ALIAS ${PROJECT_NAME})
    export_module(${PROJECT_NAME})
else ()
    message(ERROR "'openvino_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
