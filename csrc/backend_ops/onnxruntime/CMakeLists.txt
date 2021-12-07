# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_onnxruntime_ops)

include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)
set_targets(${PROJECT_NAME} ORT_OPS_OBJ ORT_OPS_STATIC ORT_OPS_MODULE)

# add plugin source
file(GLOB_RECURSE ORT_OPS_SRCS *.cpp)
build_object_target(${ORT_OPS_OBJ} "${ORT_OPS_SRCS}")
target_include_directories(${ORT_OPS_OBJ} PUBLIC
        $<BUILD_INTERFACE:${ONNXRUNTIME_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>)
target_link_directories(${ORT_OPS_OBJ} PUBLIC
        ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(${ORT_OPS_OBJ} PUBLIC onnxruntime)

add_library(${ORT_OPS_MODULE} MODULE $<TARGET_OBJECTS:${ORT_OPS_OBJ}>)
target_link_libraries(${ORT_OPS_MODULE} PRIVATE ${ORT_OPS_OBJ})
add_library(mmdeploy::onnxruntime::ops ALIAS ${ORT_OPS_MODULE})
install_targets(${ORT_OPS_MODULE})

if (MMDEPLOY_BUILD_SDK)
    ## Build static library. SDK's uses it to build `ort_net` module
    build_static_target(${ORT_OPS_STATIC} ${ORT_OPS_OBJ} "PRIVATE")
    add_library(mmdeploy::onnxruntime::ops::static ALIAS ${ORT_OPS_STATIC})
endif ()
