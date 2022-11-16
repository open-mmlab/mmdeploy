# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_transform)

set(SRCS
        collect.cpp
        compose.cpp
        center_crop.cpp
        three_crop.cpp
        ten_crop.cpp
        image2tensor.cpp
        default_format_bundle.cpp
        load.cpp
        normalize.cpp
        pad.cpp
        resize.cpp
        transform.cpp
        tracer.cpp
        lift.cpp)
mmdeploy_add_module(${PROJECT_NAME} LIBRARY "${SRCS}")
target_link_libraries(${PROJECT_NAME} PRIVATE mmdeploy_operation)
target_include_directories(
        ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/preprocess>)
add_library(mmdeploy::transform ALIAS ${PROJECT_NAME})
