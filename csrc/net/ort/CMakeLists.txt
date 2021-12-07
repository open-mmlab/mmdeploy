# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ort_net)

if ("cpu" IN_LIST MMDEPLOY_TARGET_DEVICES)
    include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

    set_targets(${PROJECT_NAME} ORT_NET_OBJ ORT_NET_STATIC ORT_NET_SHARED)

    build_object_target(${ORT_NET_OBJ} ort_net.cpp)
    target_include_directories(${ORT_NET_OBJ} PUBLIC ${ONNXRUNTIME_DIR}/include)
    target_link_directories(${ORT_NET_OBJ} PUBLIC ${ONNXRUNTIME_DIR}/lib)
    target_link_libraries(${ORT_NET_OBJ} PRIVATE mmdeploy::core::static
            PUBLIC onnxruntime)

    build_static_target(${ORT_NET_STATIC} ${ORT_NET_OBJ} "PUBLIC")
    add_library(mmdeploy::ort_net::static ALIAS ${ORT_NET_STATIC})

    build_shared_target(${ORT_NET_SHARED} ${ORT_NET_OBJ} "PUBLIC")
    target_link_libraries(${ORT_NET_SHARED} PRIVATE
            -Wl,--whole-archive
            mmdeploy::onnxruntime::ops::static
            -Wl,--no-whole-archive)
    add_library(mmdeploy::ort_net ALIAS ${ORT_NET_SHARED})

    export_module(${ORT_NET_STATIC} ${ORT_NET_SHARED} ${ORT_NET_OBJ})
    export_module(mmdeploy_onnxruntime_ops_static "" mmdeploy_onnxruntime_ops_obj)
else ()
    message(ERROR "'ort_net' is NOT supported in target devices: ${MMDEPLOY_TARGET_DEVICES}")
endif ()
