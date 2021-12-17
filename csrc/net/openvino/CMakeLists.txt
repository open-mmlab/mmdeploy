# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_openvino_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

    set_targets(${PROJECT_NAME} OPENVINO_NET_OBJ OPENVINO_NET_STATIC OPENVINO_NET_SHARED)

    find_package(InferenceEngine REQUIRED)

    build_object_target(${OPENVINO_NET_OBJ} openvino_net.cpp)
    target_link_libraries(${OPENVINO_NET_OBJ} PRIVATE
            mmdeploy::core::static
            ${InferenceEngine_LIBRARIES})

    build_static_target(${OPENVINO_NET_STATIC} ${OPENVINO_NET_OBJ} "PUBLIC")
    add_library(mmdeploy::openvino_net::static ALIAS ${OPENVINO_NET_STATIC})

    build_shared_target(${OPENVINO_NET_SHARED} ${OPENVINO_NET_OBJ} "PRIVATE")
    add_library(mmdeploy::openvino_net ALIAS ${OPENVINO_NET_SHARED})

    export_module(${OPENVINO_NET_STATIC} ${OPENVINO_NET_SHARED} ${OPENVINO_NET_OBJ})
else ()
    message(ERROR "'openvino_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
