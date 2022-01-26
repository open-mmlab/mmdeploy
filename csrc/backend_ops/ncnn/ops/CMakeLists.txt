# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy_ncnn_ops)

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)

# add plugin source
file(GLOB_RECURSE NCNN_OPS_SRCS *.cpp)
add_library(${PROJECT_NAME}_obj OBJECT "${NCNN_OPS_SRCS}")
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_link_libraries(${PROJECT_NAME}_obj PUBLIC ncnn)
mmdeploy_export(${PROJECT_NAME}_obj)

mmdeploy_add_module(${PROJECT_NAME} SHARED "")
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_obj)
add_library(mmdeploy::ncnn_ops ALIAS ${PROJECT_NAME})


if (MMDEPLOY_BUILD_SDK)
    ## Build static library. SDK's uses it to build `ncnn_net` module
    mmdeploy_add_module(${PROJECT_NAME}_static STATIC "")
    target_link_libraries(${PROJECT_NAME}_static PRIVATE ${PROJECT_NAME}_obj)
    add_library(mmdeploy::ncnn_ops::static ALIAS ${PROJECT_NAME}_static)
endif ()
